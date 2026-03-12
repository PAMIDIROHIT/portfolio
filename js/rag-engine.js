/**
 * RAG Engine - Retrieval-Augmented Generation from Scratch
 * Implements query rewriting + hybrid retrieval (dense + BM25) + RRF fusion
 * and ColBERT-style reranking for grounded answer generation with sources
 * Pure JavaScript, no external dependencies
 */

class RAGEngine {
  constructor(documents) {
    this.documents = documents;
    this.docById = new Map(documents.map(doc => [doc.id, doc]));

    // Retrieval config tuned for in-browser responsiveness.
    this.config = {
      chunkSize: 82,
      chunkOverlap: 18,
      minChunkSize: 34,
      sentenceOverlap: 1,
      denseDim: 192,
      tokenDim: 32,
      sparseTopK: 42,
      denseTopK: 42,
      fusedTopK: 32,
      rerankTopK: 18,
      rrfK: 60,
      minScore: 0.01,
    };

    // Chunk corpus and retrieval indexes.
    this.chunks = [];
    this.chunkTermFreq = [];
    this.chunkLengths = [];
    this.avgChunkLength = 0;
    this.termDocFreq = new Map();
    this.bm25Idf = new Map();

    this.chunkDenseVectors = [];
    this.chunkTokenVectors = [];
    this.docChunkMap = new Map();
    this.chunkIndexById = new Map();

    this._lastTopic = null;       // Tracks last response for follow-ups
    this._lastRetrievalTrace = null;
    this._buildIndex();
  }

  // ─── Tokenizer ────────────────────────────────────────────────────────────

  _tokenize(text) {
    return text
      .toLowerCase()
      .replace(/[^a-z0-9\s\.\+#]/g, ' ')
      .split(/\s+/)
      .filter(t => t.length > 1 && !this._isStopword(t));
  }

  _isStopword(word) {
    const STOP = new Set([
      'a','an','the','and','or','but','in','on','at','to','for','of','with',
      'by','from','is','are','was','were','be','been','being','have','has',
      'had','do','does','did','will','would','could','should','may','might',
      'shall','can','this','that','these','those','i','me','my','myself','we',
      'our','you','your','he','his','she','her','it','its','they','them','their',
      'what','which','who','whom','when','where','why','how','all','any','both',
      'each','few','more','most','other','some','such','no','not','only','same',
      'so','than','too','very','just','also','as','s','t','don','about'
    ]);
    return STOP.has(word);
  }

  _getDocText(doc) {
    return `${doc.title} ${doc.title} ${doc.content} ${doc.keywords.join(' ')}`;
  }

  _wordCount(text) {
    return String(text || '').split(/\s+/).filter(Boolean).length;
  }

  _trimWords(text, maxWords) {
    const words = String(text || '').split(/\s+/).filter(Boolean);
    if (words.length <= maxWords) return words.join(' ');
    return `${words.slice(0, maxWords).join(' ')}...`;
  }

  _splitSentences(text) {
    const normalized = String(text || '').replace(/\s+/g, ' ').trim();
    if (!normalized) return [];
    const parts = normalized.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [normalized];
    return parts.map(part => part.trim()).filter(Boolean);
  }

  _targetChunkSizeFor(doc) {
    const byCategory = {
      contact: 34,
      personal: 46,
      education: 46,
      achievement: 52,
      conversational: 52,
      skills: 58,
      project: 84,
    };
    return byCategory[doc.category] || this.config.chunkSize;
  }

  _chunkTypeBoost(type) {
    return {
      summary: 1.05,
      passage: 1.0,
      keyword: 1.01,
      qa: 0.98,
      dialogue: 0.96,
    }[type] || 1;
  }

  _buildContextualPrefix(doc, summary, type) {
    const cueByType = {
      summary: 'High-level summary',
      passage: 'Detailed evidence',
      keyword: 'Entity and keyword index',
      qa: 'Question answering support',
      dialogue: 'Synthetic user-assistant training pair',
    };
    const keywords = (doc.keywords || []).slice(0, 8).join(', ');
    return `${doc.title}. ${cueByType[type] || 'Document context'}. Category: ${doc.category}. Summary: ${summary}. Key topics: ${keywords}.`;
  }

  _scoreChunkQuality(chunkText, doc, type, targetSize) {
    const words = String(chunkText || '').split(/\s+/).filter(Boolean);
    const wordCount = words.length || 1;
    const uniqueRatio = new Set(words.map(word => word.toLowerCase())).size / wordCount;
    const lower = String(chunkText || '').toLowerCase();
    const keywordHits = (doc.keywords || [])
      .slice(0, 10)
      .filter(keyword => lower.includes(String(keyword).toLowerCase()))
      .length;
    const lengthScore = Math.max(0.35, 1 - Math.abs(wordCount - targetSize) / Math.max(targetSize, 1));
    const keywordScore = Math.min(1, keywordHits / Math.max(1, Math.min(5, (doc.keywords || []).length)));
    const boundaryScore =
      (/^[A-Z0-9]/.test(String(chunkText || '').trim()) ? 0.5 : 0.3) +
      (/[.!?]$/.test(String(chunkText || '').trim()) ? 0.5 : 0.3);
    const densityScore = Math.min(1, uniqueRatio / 0.72);
    const quality = (0.34 * lengthScore) + (0.28 * keywordScore) + (0.20 * boundaryScore) + (0.18 * densityScore);
    return Number((quality * this._chunkTypeBoost(type)).toFixed(3));
  }

  _createChunk(doc, rawText, type, localChunkIndex, meta = {}) {
    const summary = meta.summary || this._trimWords(doc.content, 28);
    const contextualText = `${this._buildContextualPrefix(doc, summary, type)} ${String(rawText || '').trim()}`.trim();
    return {
      id: `${doc.id}#${localChunkIndex}`,
      docId: doc.id,
      title: doc.title,
      category: doc.category,
      rawText: String(rawText || '').trim(),
      text: contextualText,
      type,
      source: this._sourceForDoc(doc.id),
      quality: this._scoreChunkQuality(rawText, doc, type, this._targetChunkSizeFor(doc)),
      ...meta,
    };
  }

  _buildSyntheticDialogueChunks(doc, localChunkIndex, summary, focusSentence) {
    const title = doc.title;
    const keywords = (doc.keywords || []).slice(0, 8).join(', ');
    const prompts = [];

    switch (doc.category) {
      case 'project':
        prompts.push(
          `What is ${title}?`,
          `How does ${title} work?`,
          `Which technologies are used in ${title}?`,
          `Why is ${title} impressive in Rohit's portfolio?`
        );
        break;
      case 'skills':
        prompts.push(
          `What skills does Rohit have in ${title}?`,
          `Which tools and technologies are covered in ${title}?`,
          `Give a recruiter-friendly summary of Rohit's expertise in ${title}.`
        );
        break;
      case 'contact':
        prompts.push(
          `How can I contact Rohit?`,
          `What is Rohit's email and LinkedIn?`,
          `Is Rohit open to internships or full-time roles?`
        );
        break;
      case 'achievement':
        prompts.push(
          `What achievement does Rohit have in ${title}?`,
          `Why does ${title} matter?`,
          `Give a short achievement summary for ${title}.`
        );
        break;
      case 'education':
        prompts.push(
          `Where did Rohit study?`,
          `What is Rohit's academic background?`,
          `What are the key facts in ${title}?`
        );
        break;
      default:
        prompts.push(
          `Tell me about ${title}.`,
          `Explain ${title} in simple terms.`,
          `What are the key highlights of ${title}?`
        );
        break;
    }

    return prompts.map((question, offset) => this._createChunk(
      doc,
      `User question: ${question} Assistant answer: ${summary}${focusSentence ? ` Key detail: ${focusSentence}` : ''}${keywords ? ` Topics: ${keywords}.` : ''}`,
      'dialogue',
      localChunkIndex + offset,
      { summary, syntheticQuestion: question }
    ));
  }

  _sourceForDoc(docId) {
    const urlMap = (typeof PROJECT_URLS !== 'undefined') ? PROJECT_URLS : {};
    if (urlMap[docId]) return urlMap[docId];
    if (typeof PROFILE !== 'undefined' && PROFILE.github) return PROFILE.github;
    return '';
  }

  _hash(value, seed = 2166136261) {
    let h = seed >>> 0;
    for (let i = 0; i < value.length; i++) {
      h ^= value.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  }

  _normalizeDenseVector(vec) {
    let norm = 0;
    for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < vec.length; i++) vec[i] /= norm;
    return vec;
  }

  _dotDense(vecA, vecB) {
    const len = Math.min(vecA.length, vecB.length);
    let dot = 0;
    for (let i = 0; i < len; i++) dot += vecA[i] * vecB[i];
    return dot;
  }

  _encodeDense(text) {
    const tokens = this._tokenize(text);
    const vec = new Float32Array(this.config.denseDim);

    if (tokens.length === 0) return vec;

    for (let i = 0; i < tokens.length; i++) {
      const token = tokens[i];
      const idfBoost = this.bm25Idf.get(token) || 1;

      const h1 = this._hash(token, 2166136261);
      const h2 = this._hash(token, 1469598103);
      const idx1 = h1 % this.config.denseDim;
      const idx2 = h2 % this.config.denseDim;
      const sign1 = ((h1 >>> 1) & 1) ? 1 : -1;
      const sign2 = ((h2 >>> 1) & 1) ? 1 : -1;

      vec[idx1] += sign1 * (1 + 0.25 * idfBoost);
      vec[idx2] += sign2 * 0.45;

      if (i < tokens.length - 1) {
        const bigram = `${token}_${tokens[i + 1]}`;
        const hb = this._hash(bigram, 709607);
        const idxB = hb % this.config.denseDim;
        vec[idxB] += (((hb >>> 1) & 1) ? 0.35 : -0.35);
      }
    }

    return this._normalizeDenseVector(vec);
  }

  _encodeTokenVector(token) {
    const vec = new Float32Array(this.config.tokenDim);
    const lower = token.toLowerCase();
    const grams = [];

    if (lower.length < 3) {
      grams.push(lower);
    } else {
      for (let i = 0; i <= lower.length - 3; i++) {
        grams.push(lower.slice(i, i + 3));
      }
    }

    for (const gram of grams) {
      const h = this._hash(gram, 32452843);
      const idx = h % this.config.tokenDim;
      vec[idx] += ((h & 1) ? 1 : -1);
    }

    return this._normalizeDenseVector(vec);
  }

  _buildChunks() {
    this.chunks = [];
    this.docChunkMap = new Map();
    this.chunkIndexById = new Map();

    this.documents.forEach(doc => {
      const sentences = this._splitSentences(doc.content);
      const summary = this._trimWords(sentences.slice(0, 2).join(' '), 38);
      const focusSentence = this._trimWords(sentences[2] || sentences[1] || sentences[0] || doc.content, 26);
      const targetSize = this._targetChunkSizeFor(doc);

      let localChunkIndex = 0;

      this.chunks.push(this._createChunk(
        doc,
        summary,
        'summary',
        localChunkIndex++,
        { summary, sentenceStart: 0, sentenceEnd: Math.min(1, Math.max(0, sentences.length - 1)) }
      ));

      const keywordLine = (doc.keywords || []).slice(0, 16).join(', ');
      if (keywordLine) {
        this.chunks.push(this._createChunk(
          doc,
          `Technologies, entities, and topics for ${doc.title}: ${keywordLine}.`,
          'keyword',
          localChunkIndex++,
          { summary }
        ));
      }

      if (sentences.length === 0) {
        this.chunks.push(this._createChunk(doc, doc.content, 'passage', localChunkIndex++, { summary }));
      } else {
        const sentenceWordCounts = sentences.map(sentence => this._wordCount(sentence));
        let start = 0;

        while (start < sentences.length) {
          let end = start;
          let words = 0;

          while (end < sentences.length) {
            const nextWords = sentenceWordCounts[end] || 0;
            if (end > start && words >= this.config.minChunkSize && words + nextWords > targetSize) break;
            words += nextWords;
            end += 1;
            if (words >= targetSize) break;
          }

          const rawText = sentences.slice(start, end).join(' ').trim();
          if (rawText) {
            this.chunks.push(this._createChunk(
              doc,
              rawText,
              'passage',
              localChunkIndex++,
              { summary, sentenceStart: start, sentenceEnd: end - 1 }
            ));
          }

          if (end >= sentences.length) break;
          start = Math.max(start + 1, end - this.config.sentenceOverlap);
        }
      }

      this.chunks.push(this._createChunk(
        doc,
        `Question: What is ${doc.title}? Answer: ${summary}`,
        'qa',
        localChunkIndex++,
        { summary }
      ));

      this.chunks.push(this._createChunk(
        doc,
        `Question: Which technologies or topics are related to ${doc.title}? Answer: ${keywordLine}. ${focusSentence}`,
        'qa',
        localChunkIndex++,
        { summary }
      ));

      const dialogueChunks = this._buildSyntheticDialogueChunks(doc, localChunkIndex, summary, focusSentence);
      dialogueChunks.forEach(chunk => this.chunks.push(chunk));
    });

    this.chunks.forEach((chunk, index) => {
      this.chunkIndexById.set(chunk.id, index);
      const existing = this.docChunkMap.get(chunk.docId) || [];
      existing.push(index);
      this.docChunkMap.set(chunk.docId, existing);
    });
  }

  // ─── Index Building ────────────────────────────────────────────────────────

  _buildIndex() {
    this._buildChunks();
    this._buildSparseIndex();
    this._buildDenseIndex();
  }

  _buildSparseIndex() {
    this.chunkTermFreq = [];
    this.chunkLengths = [];
    this.termDocFreq = new Map();

    for (let i = 0; i < this.chunks.length; i++) {
      const tokens = this._tokenize(this.chunks[i].text);
      const tf = new Map();

      for (const token of tokens) {
        tf.set(token, (tf.get(token) || 0) + 1);
      }

      this.chunkTermFreq[i] = tf;
      this.chunkLengths[i] = tokens.length;

      const seen = new Set(tokens);
      for (const token of seen) {
        this.termDocFreq.set(token, (this.termDocFreq.get(token) || 0) + 1);
      }
    }

    const totalLength = this.chunkLengths.reduce((acc, len) => acc + len, 0);
    this.avgChunkLength = totalLength / Math.max(1, this.chunkLengths.length);

    const N = this.chunks.length;
    this.bm25Idf = new Map();
    for (const [term, df] of this.termDocFreq.entries()) {
      const idf = Math.log(1 + ((N - df + 0.5) / (df + 0.5)));
      this.bm25Idf.set(term, idf);
    }
  }

  _buildDenseIndex() {
    this.chunkDenseVectors = this.chunks.map(chunk => this._encodeDense(chunk.text));

    this.chunkTokenVectors = this.chunks.map(chunk => {
      const tokens = this._tokenize(chunk.text).slice(0, 48);
      return tokens.map(token => this._encodeTokenVector(token));
    });
  }

  // ─── Query Rewriting + Hybrid Retrieval Helpers ───────────────────────────

  _rewriteQuery(query) {
    const normalized = (query || '').toLowerCase().replace(/\s+/g, ' ').trim();

    const typoMap = {
      hellow: 'hello',
      repsoniveness: 'responsiveness',
      repsonse: 'response',
      machin: 'machine',
      learnig: 'learning',
      genaii: 'genai',
    };

    const expansionMap = {
      ml: 'machine learning',
      dl: 'deep learning',
      genai: 'generative ai',
      llm: 'large language model',
      rag: 'retrieval augmented generation',
      cv: 'computer vision',
      nlp: 'natural language processing',
      js: 'javascript',
      ts: 'typescript',
    };

    const correctedTokens = normalized
      .split(/\s+/)
      .map(token => typoMap[token] || token)
      .filter(Boolean);

    const expandedTokens = [];
    correctedTokens.forEach(token => {
      if (expansionMap[token]) {
        expandedTokens.push(...expansionMap[token].split(' '));
      } else {
        expandedTokens.push(token);
      }
    });

    const rewrites = new Set();
    const corrected = correctedTokens.join(' ').trim();
    const expanded = expandedTokens.join(' ').trim();

    rewrites.add(normalized);
    if (corrected && corrected !== normalized) rewrites.add(corrected);
    if (expanded && expanded !== corrected) rewrites.add(expanded);

    if (/web|frontend|backend|full.?stack|react|node|javascript|typescript|html|css/.test(expanded)) {
      rewrites.add(`${expanded} web development frontend backend react nodejs javascript typescript`);
    }
    if (/ml|machine learning|deep learning|ai|model|pytorch|xgboost|scikit/.test(expanded)) {
      rewrites.add(`${expanded} machine learning deep learning models training evaluation pytorch scikit-learn`);
    }
    if (/genai|generative|llm|rag|agentic|langchain|langgraph|prompt|embedding/.test(expanded)) {
      rewrites.add(`${expanded} generative ai llm rag embeddings retrieval langchain langgraph agents`);
    }

    const rewriteList = Array.from(rewrites)
      .filter(text => text && text.length > 1)
      .slice(0, 4)
      .map((text, index) => ({
        text,
        weight: [1.0, 0.88, 0.76, 0.68][index] || 0.6,
      }));

    return {
      primary: rewriteList[0] ? rewriteList[0].text : normalized,
      rewrites: rewriteList,
    };
  }

  _bm25Score(tf, dl, idf, k1 = 1.2, b = 0.75) {
    const denom = tf + k1 * (1 - b + b * (dl / Math.max(1, this.avgChunkLength)));
    return idf * ((tf * (k1 + 1)) / (denom || 1));
  }

  _bm25Retrieve(rewrites, topK = this.config.sparseTopK) {
    const scoreMap = new Map();

    rewrites.forEach(rewrite => {
      const qTokens = this._tokenize(rewrite.text);
      const qtf = new Map();
      qTokens.forEach(token => qtf.set(token, (qtf.get(token) || 0) + 1));

      for (const [term, qFreq] of qtf.entries()) {
        const idf = this.bm25Idf.get(term);
        if (!idf) continue;

        for (let i = 0; i < this.chunks.length; i++) {
          const tf = this.chunkTermFreq[i].get(term) || 0;
          if (tf === 0) continue;
          const dl = this.chunkLengths[i] || 1;
          const queryWeight = 1 + Math.log(1 + qFreq);
          const qualityBoost = (this.chunks[i] ? this.chunks[i].quality : 1) * this._chunkTypeBoost(this.chunks[i] ? this.chunks[i].type : 'passage');
          const contribution = rewrite.weight * queryWeight * this._bm25Score(tf, dl, idf) * qualityBoost;
          scoreMap.set(i, (scoreMap.get(i) || 0) + contribution);
        }
      }
    });

    return Array.from(scoreMap.entries())
      .map(([chunkIndex, score]) => ({ chunkIndex, score }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  _denseRetrieve(rewrites, topK = this.config.denseTopK) {
    const scoreMap = new Map();

    rewrites.forEach(rewrite => {
      const qVec = this._encodeDense(rewrite.text);
      for (let i = 0; i < this.chunkDenseVectors.length; i++) {
        const sim = this._dotDense(qVec, this.chunkDenseVectors[i]);
        if (sim <= 0) continue;
        const qualityBoost = (this.chunks[i] ? this.chunks[i].quality : 1) * this._chunkTypeBoost(this.chunks[i] ? this.chunks[i].type : 'passage');
        scoreMap.set(i, (scoreMap.get(i) || 0) + (rewrite.weight * sim * qualityBoost));
      }
    });

    return Array.from(scoreMap.entries())
      .map(([chunkIndex, score]) => ({ chunkIndex, score }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  _reciprocalRankFusion(sparse, dense, topK = this.config.fusedTopK) {
    const sparseRank = new Map();
    const denseRank = new Map();
    const sparseScore = new Map();
    const denseScore = new Map();

    sparse.forEach((item, index) => {
      sparseRank.set(item.chunkIndex, index + 1);
      sparseScore.set(item.chunkIndex, item.score);
    });
    dense.forEach((item, index) => {
      denseRank.set(item.chunkIndex, index + 1);
      denseScore.set(item.chunkIndex, item.score);
    });

    const chunkSet = new Set([...sparseRank.keys(), ...denseRank.keys()]);
    const fused = [];

    chunkSet.forEach(chunkIndex => {
      const rankS = sparseRank.get(chunkIndex);
      const rankD = denseRank.get(chunkIndex);
      const qualityBoost = (this.chunks[chunkIndex] ? this.chunks[chunkIndex].quality : 1) * this._chunkTypeBoost(this.chunks[chunkIndex] ? this.chunks[chunkIndex].type : 'passage');
      const rrf =
        (rankS ? 1 / (this.config.rrfK + rankS) : 0) +
        (rankD ? 1 / (this.config.rrfK + rankD) : 0);

      fused.push({
        chunkIndex,
        rrfScore: rrf * qualityBoost,
        sparseRank: rankS || null,
        denseRank: rankD || null,
        sparseScore: sparseScore.get(chunkIndex) || 0,
        denseScore: denseScore.get(chunkIndex) || 0,
      });
    });

    return fused
      .sort((a, b) => b.rrfScore - a.rrfScore)
      .slice(0, topK);
  }

  _colbertRerank(query, fused, topK = this.config.rerankTopK) {
    const queryTokens = this._tokenize(query).slice(0, 24);
    if (queryTokens.length === 0) {
      return fused.slice(0, topK).map(item => ({
        ...item,
        colbertScore: 0,
        finalScore: item.rrfScore,
      }));
    }

    const qTokenVecs = queryTokens.map(token => this._encodeTokenVector(token));

    const reranked = fused.slice(0, this.config.fusedTopK).map(item => {
      const dTokenVecs = this.chunkTokenVectors[item.chunkIndex] || [];
      const quality = this.chunks[item.chunkIndex] ? this.chunks[item.chunkIndex].quality : 1;
      let colbertScore = 0;

      if (dTokenVecs.length > 0) {
        for (const qVec of qTokenVecs) {
          let maxSim = 0;
          for (const dVec of dTokenVecs) {
            const sim = this._dotDense(qVec, dVec);
            if (sim > maxSim) maxSim = sim;
          }
          colbertScore += Math.max(0, maxSim);
        }
        colbertScore /= qTokenVecs.length;
      }

      // Scale RRF into a range comparable with late-interaction score.
      const scaledRrf = item.rrfScore * 20;
      const finalScore = 0.64 * colbertScore + 0.24 * scaledRrf + 0.12 * quality;

      return {
        ...item,
        colbertScore,
        finalScore,
      };
    });

    return reranked
      .sort((a, b) => b.finalScore - a.finalScore)
      .slice(0, topK);
  }

  _aggregateDocResults(reranked, topK) {
    const docMap = new Map();

    reranked.forEach(item => {
      const chunk = this.chunks[item.chunkIndex];
      if (!chunk) return;
      const doc = this.docById.get(chunk.docId);
      if (!doc) return;

      const existing = docMap.get(doc.id);
      if (!existing) {
        docMap.set(doc.id, {
          doc,
          score: item.finalScore,
          chunk,
          source: chunk.source,
          retrieval: item,
          evidence: [chunk],
          bestScore: item.finalScore,
        });
        return;
      }

      existing.score += item.finalScore * 0.32;

      if (item.finalScore > existing.bestScore) {
        existing.bestScore = item.finalScore;
        existing.chunk = chunk;
        existing.source = chunk.source;
        existing.retrieval = item;
      }
      if (existing.evidence.length < 3 && !existing.evidence.some(c => c.id === chunk.id)) {
        existing.evidence.push(chunk);
      }
    });

    return Array.from(docMap.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  // ─── Retrieval ────────────────────────────────────────────────────────────

  retrieve(query, topK = 4) {
    const rewritten = this._rewriteQuery(query);
    const sparse = this._bm25Retrieve(rewritten.rewrites, this.config.sparseTopK);
    const dense = this._denseRetrieve(rewritten.rewrites, this.config.denseTopK);
    const fused = this._reciprocalRankFusion(sparse, dense, this.config.fusedTopK);
    const reranked = this._colbertRerank(rewritten.primary, fused, this.config.rerankTopK);

    const docResults = this._aggregateDocResults(reranked, Math.max(topK, 8));
    const queryLower = query.toLowerCase().trim();
    const isTechnicalQuery = /implement|architecture|pipeline|bge|bm25|rrf|colbert|retrieval|rag|embedding|semantic|hybrid|ml|genai|llm/.test(queryLower);

    // Lexical boost to preserve precision for exact intents.
    docResults.forEach(result => {
      const kw = result.doc.keywords.join(' ').toLowerCase();
      if (queryLower && (kw.includes(queryLower) || queryLower.includes(result.doc.category))) {
        result.score += 0.12;
      }
      if (queryLower && result.doc.title.toLowerCase().includes(queryLower)) {
        result.score += 0.2;
      }

      if (isTechnicalQuery) {
        if (result.doc.category === 'conversational') result.score -= 0.25;
        if (result.doc.category === 'project' || result.doc.category === 'skills' || result.doc.category === 'research') {
          result.score += 0.12;
        }
      }
    });

    const finalResults = docResults
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .filter(item => item.score > this.config.minScore);

    this._lastRetrievalTrace = {
      rewritten,
      sparseTop: sparse.slice(0, 5),
      denseTop: dense.slice(0, 5),
      fusedTop: fused.slice(0, 5),
      rerankedTop: reranked.slice(0, 5),
    };

    return finalResults;
  }

  // ─── Response Generation ──────────────────────────────────────────────────

  generateResponse(query, profile) {
    profile = profile || (typeof PROFILE !== 'undefined' ? PROFILE : {});
    const rawQueryLower = query.toLowerCase().trim();
    const queryLower = this._normalizeCasualText(rawQueryLower);

    // ─── Conversational Handler (catches ALL casual/social queries) ───
    const conversational = this._handleConversational(queryLower, profile);
    if (conversational) return conversational;

    // ─── Detail follow-up detection ───────────────────────────────────
    const wantsDetail = /more|detail|elaborate|explain more|tell me more|full|in.?depth|expand|go deeper|everything/.test(queryLower);
    if (wantsDetail && this._lastTopic) {
      const detail = this._detailedResponse(this._lastTopic, profile);
      return this._attachSources(detail, this._lastTopic.results || []);
    }

    // ─── Single-word / short-form query handler ────────────────────
    const shortAnswer = this._handleShortQuery(queryLower, profile);
    if (shortAnswer) return shortAnswer;

    const results = this.retrieve(queryLower || query, 6);

    if (results.length === 0) {
      return this._smartFallback(query, profile);
    }

    // Intent detection
    const isAbout     = /who|about|yourself|introduce|tell me|overview|summary|rohit|pamidi/.test(queryLower);
    const isProject   = /project|build|develop|creat|work|app|application|github|repo/.test(queryLower);
      const isSkill     = /skill|language|technolog|expertise|proficien|stack|tool|framework|frontend|backend|full.?stack/.test(queryLower);
    const isEducation = /educat|college|universit|degree|study|cgpa|gpa|iiit|btech|school|percent/.test(queryLower);
    const isContact   = /contact|email|phone|linkedin|whatsapp|reach|connect|hire|message/.test(queryLower);
    const isAchieve   = /achieve|award|leetcode|hackathon|competi|solve|problem|bah|isro|lunar/.test(queryLower);
    const isHackathon = /hackathon|bah25|bah|lunar|moon|chandrayaan|isro|space|pioneer|neurocore|agentic|smart.?city|kernelcrew|kernel.?crew/.test(queryLower);
    const isHire      = /hire|recruit|should i|good fit|team|worth|capable|can he|can rohit|is he|is rohit|recommend/.test(queryLower);
    const isFunFact   = /fun|interesting|cool|amazing|wow|surprising|random|trivia|did you know|unique|fact/.test(queryLower);
    const explicitSkillIntent = /\bskill\b|\bskills\b|tech stack|expertise|technolog/.test(queryLower);

    // Route to specific formatters
    const topDocs = results.map(r => r.doc);

    if (isContact) {
      this._lastTopic = { type: 'contact', docs: topDocs, results };
      return this._attachSources(this._formatContact(profile), results);
    }
    if (isHire) {
      this._lastTopic = { type: 'hire', docs: topDocs, results };
      return this._attachSources(this._formatHire(profile), results);
    }
    if (isHackathon) {
      this._lastTopic = { type: 'hackathon', docs: topDocs, results };
      return this._attachSources(this._formatHackathon(topDocs, queryLower), results);
    }
    if (isAbout && !isProject && !isSkill) {
      this._lastTopic = { type: 'about', docs: topDocs, results };
      return this._attachSources(this._formatAbout(profile, topDocs), results);
    }
    if (isEducation) {
      this._lastTopic = { type: 'education', docs: topDocs, results };
      return this._attachSources(this._formatEducation(profile, topDocs), results);
    }
    if (isAchieve) {
      this._lastTopic = { type: 'achievements', docs: topDocs, results };
      return this._attachSources(this._formatAchievements(topDocs), results);
    }
    if (isFunFact) {
      this._lastTopic = { type: 'funfact', docs: topDocs, results };
      return this._formatFunFact(profile);
    }
    if (isSkill && explicitSkillIntent) {
      this._lastTopic = { type: 'skills', docs: topDocs, results };
      return this._attachSources(this._formatSkills(profile, topDocs, queryLower), results);
    }
    if (isProject) {
      this._lastTopic = { type: 'projects', docs: topDocs, results };
      return this._attachSources(this._formatProjects(topDocs, queryLower, results), results);
    }
    if (isSkill) {
      this._lastTopic = { type: 'skills', docs: topDocs, results };
      return this._attachSources(this._formatSkills(profile, topDocs, queryLower), results);
    }

    // Default: use top retrieved context
    this._lastTopic = { type: 'general', docs: topDocs, results };
    return this._attachSources(this._formatGeneral(topDocs, query, results), results);
  }

  // ─── Fuzzy Match Helper ─────────────────────────────────────────
  _normalizeCasualText(text) {
    const replacements = {
      gud: 'good',
      gd: 'good',
      gm: 'gm',
      gn: 'gn',
      ga: 'good afternoon',
      ge: 'good evening',
      mrng: 'morning',
      mrning: 'morning',
      mornin: 'morning',
      evng: 'evening',
      eve: 'evening',
      nyt: 'night',
      nite: 'night',
      nitee: 'night',
      ngt: 'night',
      hlo: 'hello',
      heloo: 'hello',
      helooo: 'hello',
      sup: 'sup',
      yo: 'yo',
      thx: 'thanks',
      tysm: 'thank you',
      tc: 'take care',
    };

    return String(text || '')
      .toLowerCase()
      .replace(/[_-]+/g, ' ')
      .replace(/\s+/g, ' ')
      .trim()
      .split(' ')
      .filter(Boolean)
      .map(token => replacements[token] || token)
      .join(' ')
      .trim();
  }

  _fuzzyMatch(input, targets) {
    // Levenshtein distance for typo tolerance
    const _lev = (a, b) => {
      const m = a.length, n = b.length;
      const dp = Array.from({length: m + 1}, (_, i) => [i]);
      for (let j = 1; j <= n; j++) dp[0][j] = j;
      for (let i = 1; i <= m; i++)
        for (let j = 1; j <= n; j++)
          dp[i][j] = Math.min(
            dp[i-1][j] + 1,
            dp[i][j-1] + 1,
            dp[i-1][j-1] + (a[i-1] !== b[j-1] ? 1 : 0)
          );
      return dp[m][n];
    };
    const words = input.replace(/[^a-z\s]/g, '').split(/\s+/).filter(w => w.length > 1);
    for (const word of words) {
      for (const target of targets) {
        if (_lev(word, target) <= Math.max(1, Math.floor(target.length / 4))) return true;
      }
    }
    return false;
  }

  // ─── Comprehensive Conversational Handler ──────────────────────

  _handleConversational(q, profile) {
    const stripped = q.replace(/[^a-z0-9\s]/g, '').trim();
    const words = stripped.split(/\s+/);
    const wordCount = words.length;

    // ── 0. EMPTY / TOO SHORT / GIBBERISH (check first) ──
    // Allow 2-char abbreviations (gn, gm, hi, yo) — only block truly empty or single-char noise
    if (stripped.length < 2 || /^[^a-z]*$/i.test(stripped)) {
      return `I didn't quite catch that! 😅 Try asking me something like:\n- "Who is Rohit?"\n- "What projects has he built?"\n- "What are his skills?"\n- "How to contact Rohit?"`;
    }

    // ── 1. GOOD MORNING / EVENING / NIGHT (specific, check before fuzzy greetings) ──
    // Covers informal "good night" farewells: gud nyt, gn, nite, nyt, goodnight, g9, etc.
    if (/^(good\s*night|goodnight|gud\s*(night|nyt|nite|ngt|nite)|gd\s*(night|nyt|nite)|nyt|nite|nitey|g9|gn)([!.,?\s]*)$/i.test(q)) {
      const byes = [
        `Good night! 🌙 Sleep well! Come back anytime to explore Rohit's portfolio. If you'd like to connect, reach him at **rohithtnsp@gmail.com** or [LinkedIn](${profile.linkedin}).`,
        `Good night! 🌙 Sweet dreams! Rohit's portfolio will be here whenever you're ready. Take care!`,
        `Good night! 🌙 Bye for now! Feel free to check out Rohit's work on [GitHub](${profile.github}) anytime.`
      ];
      return byes[Math.floor(Math.random() * byes.length)];
    }
    // Covers informal "good morning/afternoon/evening" greetings: gm, gud mrng, gd mrng, etc.
    if (/^(good\s*(morning|afternoon|evening|day)|gm|gud\s*(morning|mrng|mrning|mornin|evening|evng|afternoon|day)|gd\s*(morning|mrng|evening|afternoon|day))([!.,?\s]*)$/i.test(q)) {
        let timeLabel = 'morning';
        if (/afternoon/.test(q)) timeLabel = 'afternoon';
        else if (/evening|evng/.test(q)) timeLabel = 'evening';
        else if (/day/.test(q)) timeLabel = 'day';
        return `Good ${timeLabel}! ☀️ Welcome to **Rohit AI**.\n\nI'm ready to answer any questions about Pamidi Rohit — his projects, skills, education, achievements, or how to reach him.\n\nWhat would you like to know?`;
    }
    if (/^good (morning|afternoon|evening|night|day)[!.\s]*/i.test(q)) {
      const timeGreet = q.match(/good (\w+)/i)[1];
      return `Good ${timeGreet}! 👋 Welcome to **Rohit AI**.\n\nI'm ready to answer any questions about Pamidi Rohit — his projects, skills, education, achievements, or how to reach him.\n\nWhat would you like to know?`;
    }

    // ── 2. HOW ARE YOU / WHAT'S UP (check before fuzzy greetings) ──
    if (/^(how are you|how('s| is) it going|how do you do|how('s| is) everything|what'?s up|what'?s new|how('s| is) life|you good|are you okay|are you good|how have you been|hows it going)/i.test(q)) {
      return `I'm doing great, thanks for asking! 😊 I'm here 24/7 to answer questions about **Pamidi Rohit**.\n\nI never get tired and I know all about his 22+ projects, technical skills, and achievements. What would you like to know?`;
    }

    // ── 3. IDENTITY QUESTIONS (check before fuzzy greetings) ──
    if (/^(who are you|what are you|what('?s| is) your name|your name|are you (an? )?(ai|bot|robot|chatbot|real|human)|tell me about yourself|introduce yourself|what do you do|who made you|who built you|who created you)/i.test(q)) {
      return `I'm **Rohit AI** — a custom RAG (Retrieval-Augmented Generation) chatbot built entirely from scratch by **Pamidi Rohit**.\n\n**How I work:**\n- I rewrite the query for better recall\n- I run **hybrid retrieval**: dense (BGE-style embeddings) + sparse (BM25)\n- I fuse rankings with **RRF** and rerank with a **ColBERT-style** late interaction step\n- I generate a grounded answer with source links\n\nI'm not powered by ChatGPT or Gemini — I'm a **100% custom-built AI** that demonstrates Rohit's retrieval and NLP engineering skills.\n\nAsk me anything about Rohit!`;
    }

    // ── 3.5. HELP check (must be before fuzzy greetings since "help" fuzzy-matches "helo") ──
    if (/^(help|what can you (do|tell)|how (to |do i )?use|commands|options|features|what do you know|what questions|capabilities|what should i ask|examples|sample questions|guide me)/i.test(q)) {
      return `## How to Use Rohit AI\n\nYou can ask me **anything** about Pamidi Rohit. Here are some examples:\n\n**🧑 About Rohit**\n- "Who is Pamidi Rohit?"\n- "Tell me about his background"\n- "What are his strengths?"\n\n**💻 Projects (22+)**\n- "What are all his projects?"\n- "Tell me about the Tomato food delivery app"\n- "What ML projects has he built?"\n- "Show me his hackathon projects"\n\n**🛠 Skills**\n- "What programming languages does he know?"\n- "What is his tech stack?"\n- "Does he know Docker/AWS/MLOps?"\n\n**🎓 Education & Achievements**\n- "Where does Rohit study?"\n- "How many LeetCode problems solved?"\n- "Tell me about his ISRO hackathon"\n\n**📬 Contact & Hiring**\n- "How can I reach Rohit?"\n- "Should I hire Rohit?"\n- "Is he available for internships?"\n\n**🎲 Fun**\n- "Tell me something interesting about Rohit"\n- "Fun facts"\n\nJust type naturally — I understand casual questions, typos, and follow-ups!`;
    }

    // ── 4. GREETINGS (with typo tolerance — now after specific patterns) ──
    const greetWords = ['hello','hellow','helo','hi','hey','hii','hiii','howdy','sup','yo','heya','hola','namaste','namaskar','greetings','ola','bonjour','salut','salam','ahoy','aloha','wassup','whatsup','morning','afternoon','evening'];
    // Only fuzzy-match the FIRST word — greetings always start the sentence
    const firstWord = words[0] || '';
    if (wordCount <= 4 && firstWord.length >= 2 && this._fuzzyMatch(firstWord, greetWords)) {
      const greets = [
        `Hey there! 👋 I'm **Rohit AI**, a RAG-powered chatbot built from scratch by Pamidi Rohit.\n\nI know everything about Rohit — his **22+ projects**, **skills**, **education**, **achievements**, and more.\n\nTry asking:\n- "Who is Rohit?"\n- "What are his projects?"\n- "What skills does he have?"\n- "Tell me something interesting"\n\nWhat would you like to know?`,
        `Hello! 👋 Welcome to **Rohit AI**!\n\nI'm trained on Pamidi Rohit's entire portfolio. Ask me anything about:\n- **Projects** — 22+ apps across full-stack, ML/AI, cloud\n- **Skills** — Python, React, Node.js, PyTorch, Docker, and more\n- **Achievements** — 600+ LeetCode, ISRO hackathon, multi-agent AI\n- **Education & Contact**\n\nWhat would you like to explore?`,
        `Hi! 👋 Great to see you here!\n\nI'm **Rohit AI** — ask me anything about Pamidi Rohit. I can tell you about his projects, technical skills, hackathon experiences, education, or how to get in touch with him.\n\nWhat's on your mind?`
      ];
      return greets[Math.floor(Math.random() * greets.length)];
    }

    // ── 5. FAREWELLS ──
    if (/^(bye|goodbye|good ?bye|see you|see ya|later|take care|cya|gtg|gotta go|peace|adios|sayonara|cheers|catch you later|talk later|signing off|good night|goodnight|gn)[!.\s]*$/i.test(q)) {
      const byes = [
        `Goodbye! 👋 Thanks for chatting! If you'd like to connect with Rohit, reach him at **rohithtnsp@gmail.com** or on [LinkedIn](${profile.linkedin}). Have a great day!`,
        `See you later! 👋 Remember, you can always come back to learn more about Rohit. Feel free to reach him at **rohithtnsp@gmail.com** anytime!`,
        `Take care! 👋 If you found Rohit interesting, check out his work on [GitHub](${profile.github}) or connect on [LinkedIn](${profile.linkedin}). Bye!`
      ];
      return byes[Math.floor(Math.random() * byes.length)];
    }

    // ── 6. THANKS / APPRECIATION ──
    if (/^(thanks?|thank you|thx|ty|appreciate|appreciate it|grateful|great job|good job|nice|well done|awesome|cool|perfect|excellent|wonderful|brilliant|sick|dope|lit|fire|wonderful job|amazing answer|helpful|that('s| is| was) helpful|that('s| is| was) great|that('s| is| was) awesome|that('s| is| was) perfect|thank u|thanks a lot|much appreciated)[!.\s]*$/i.test(q)) {
      const thanks = [
        `You're welcome! 😊 Happy to help. Feel free to ask more about Rohit's projects, skills, or anything else!`,
        `Glad I could help! 🙌 If you have more questions about Rohit, I'm right here. No question is too specific or too broad!`,
        `Thanks for the kind words! 😊 There's so much more to explore — Rohit has 22+ projects. Want to hear about a specific one?`
      ];
      return thanks[Math.floor(Math.random() * thanks.length)];
    }

    // ── 7. AFFIRMATIVES / YES / NO / OK ──
    if (/^(yes|yeah|yep|yup|ya|yea|sure|ok|okay|alright|got it|understood|right|correct|exactly|true|affirmative|roger|copy|bet)[!.\s]*$/i.test(q)) {
      if (this._lastTopic) {
        return `Great! Would you like me to go into **more detail** about that, or would you like to explore something else?\n\nYou can say *"tell me more"* or ask about a different topic: **projects**, **skills**, **education**, **achievements**, or **contact info**.`;
      }
      return `Got it! 👍 What would you like to know about Rohit? I can tell you about his **projects**, **skills**, **education**, **achievements**, or how to **contact** him.`;
    }

    // ── 8. NEGATIVES ──
    if (/^(no|nah|nope|not really|no thanks|nah i'?m good|nothing|never ?mind|forget it|skip|pass)[!.\s]*$/i.test(q)) {
      return `No problem! 😊 If you change your mind, I'm here. You can ask me anything about Rohit anytime!\n\nSome popular questions:\n- "What are Rohit's projects?"\n- "Tell me something interesting"\n- "How to contact Rohit?"`;
    }

    // ── 9. FOLLOW-UPS ──
    if (/^(and\??|so\??|what else|anything else|continue|go on|next|more please|keep going|what more|elaborate|expand)[!?\s]*$/i.test(q)) {
      if (this._lastTopic) {
        const detail = this._detailedResponse(this._lastTopic, profile);
        return this._attachSources(detail, this._lastTopic.results || []);
      }
      return `Sure! What topic would you like me to explore? I can tell you about:\n- **Projects** — 22+ applications\n- **Skills** — Full-stack, ML/AI, DevOps\n- **Achievements** — LeetCode, hackathons\n- **Education** — IIIT Sri City\n- **Contact** — Email, LinkedIn, GitHub`;
    }

    // ── 10. COMPLIMENTS ──
    if (/^(you('re| are) (smart|good|great|amazing|awesome|helpful|impressive|clever|brilliant)|good (bot|ai|job)|nice (bot|ai|work)|impressive|well done|bravo|wow|amazing|smart bot)[!.\s]*$/i.test(q)) {
      return `Thank you! 😊 That means a lot. I was built from scratch by Rohit using a **multi-stage RAG pipeline**: query rewriting, dense+sparse retrieval, RRF fusion, and ColBERT-style reranking — no external AI APIs needed.\n\nThis chatbot itself is a demonstration of Rohit's **NLP** and **information retrieval** engineering skills!`;
    }

    // ── 11. HIRE / RECRUIT / SHOULD I HIRE ──
    if (/should i (hire|recruit|consider|interview|pick|choose)|worth (hiring|recruiting|interviewing)|good (fit|candidate|hire|developer|engineer)|strong candidate|why (should|hire)|convince me|sell (me|yourself)|pitch|impress me/i.test(q)) {
      return this._formatHire(profile);
    }

    // ── 12. CAN ROHIT DO X / DOES ROHIT KNOW X ──
    if (/^(can (rohit|he)|does (rohit|he) (know|have|do)|is (rohit|he) (good|experienced|skilled|proficient|familiar))/i.test(q)) {
      return this._handleAbilityQuestion(q, profile);
    }

    // ── 13. COMPARISON ──
    if (/better than|compared to|vs |versus|how does (rohit|he) (compare|stack|rank|measure)|among|relative to/i.test(q)) {
      return `I focus specifically on **Pamidi Rohit's** portfolio and achievements. Rather than comparisons, here's what makes him stand out:\n\n- **22+ production projects** spanning full-stack, ML/AI, and cloud\n- **600+ LeetCode problems** solved\n- **ISRO BAH25 hackathon** — physics-informed neural networks\n- **Multi-agent AI** — 6 specialized agents in NeuroCore\n- **Enterprise-grade code** — NestJS API with 51 unit tests\n\nWant to dive into any specific area?`;
    }

    // ── 14. FUN / JOKE / INTERESTING ──
    if (/^(tell me (a |something )?(joke|funny|fun fact|interesting|cool|random)|fun fact|something (interesting|cool|random)|surprise me|amaze me|blow my mind|what'?s (cool|interesting|special))/i.test(q)) {
      return this._formatFunFact(profile);
    }

    // ── 15. OFF-TOPIC politeness (weather, time, etc.) ──
    if (/weather|temperature|time|date|politics|cricket|football|movie|song|music|game|news/i.test(q) && !/project|build|rohit|skill/.test(q)) {
      return `That's a great question, but I'm specialized in **Pamidi Rohit's portfolio** only! 😊\n\nI can't help with general topics, but I'm an expert on:\n- Rohit's **22+ projects** (full-stack, ML/AI, cloud)\n- His **technical skills** (Python, React, Node.js, PyTorch)\n- His **achievements** (600+ LeetCode, ISRO hackathon)\n- **How to contact him**\n\nWhat would you like to know about Rohit?`;
    }

    // ── 16. META QUESTIONS (how do you work, what model) ──
    if (/what (model|algorithm|technology|method)|how do you (work|function|think|operate)|how does (your|this) (rag|pipeline|retrieval|chatbot) work|rag pipeline|retrieval pipeline|hybrid retrieval|bm25|rrf|colbert|powered by|built with|what('s| is) behind|architecture|how were you (made|built|trained|created)|your (technology|stack|source)|open.?source/i.test(q)) {
      return `**How Rohit AI Works:**\n\n1. **Knowledge Base + Chunking** — Rohit's portfolio is split into high-quality overlapping chunks plus synthetic Q/A chunks\n2. **Query Rewriting** — your prompt is expanded and normalized to improve recall\n3. **Hybrid Retrieval** — dense retrieval (BGE-style local embeddings) + sparse retrieval (BM25)\n4. **Fusion + Rerank** — Reciprocal Rank Fusion (RRF) followed by ColBERT-style late-interaction reranking\n5. **Grounded Generation** — top-K context is used to generate answers with source links\n6. **Conversational Layer** — I still handle greetings, follow-ups, typos, and casual chat naturally\n\nNo external AI APIs — this is a **custom RAG engine built from scratch** in JavaScript. The code is part of Rohit's portfolio itself!`;
    }

    // ── 17. EMOTIONAL / CONFUSED ──
    if (/confused|lost|don'?t (understand|know what|get)|help me|i'?m (stuck|frustrated|confused|lost)|not sure|what should/i.test(q)) {
      return `No worries at all! Let me help. 😊\n\nHere are some simple questions you can start with:\n\n1. **"Who is Rohit?"** — Quick overview\n2. **"What are his projects?"** — See all 22+ apps\n3. **"Tell me his skills"** — Full tech stack\n4. **"How to contact Rohit?"** — Email, LinkedIn, GitHub\n5. **"Fun facts about Rohit"** — Interesting highlights\n\nJust pick one! Or type anything naturally — I'll figure out what you mean.`;
    }

    // ── 18. FEEDBACK (positive or negative) ──
    if (/wrong|incorrect|not (right|correct|true|accurate)|bad answer|useless|terrible|garbage|stupid|dumb|broken|fix/i.test(q) && wordCount <= 8) {
      return `I'm sorry about that! 😔 I'm a rule-based RAG system, so I might sometimes miss context.\n\nCould you try rephrasing your question? For example:\n- Instead of a short query, try: *"Tell me about Rohit's ML projects"*\n- Be specific: *"What tech stack does the Tomato app use?"*\n- Or ask directly: *"What is Rohit's email?"*\n\nI'll do my best to give you a great answer!`;
    }

    // ── 19. NICE TO MEET YOU / PLEASURE ──
    if (/nice to meet|pleased to meet|pleasure|glad to (meet|be here|chat)|good to (meet|know|see)/i.test(q)) {
      return `Nice to meet you too! 😊 I'm **Rohit AI**, here to tell you everything about Pamidi Rohit.\n\nHe's a B.Tech AI student at IIIT Sri City who has built 22+ projects and solved 600+ LeetCode problems. Ask me anything!`;
    }

    // ── 21. AGE / PERSONAL QUESTIONS ──
    if (/how old|age|birthday|born|dob|date of birth/i.test(q)) {
      return `Rohit is a **B.Tech student** at IIIT Sri City (batch 2023-2027). He completed his intermediate education in June 2023 at Sri Chaitanya Junior College with 93.7%.\n\nWant to know about his education or projects?`;
    }

    // ── 22. LOCATION ──
    if (/where (is|does|from)|location|city|state|country|based|live|stay|hometown|native/i.test(q) && /rohit|he|him|his/.test(q)) {
      return `Rohit studies at **IIIT Sri City**, Andhra Pradesh, India. He completed his school education at Sri Chaitanya Junior College.\n\nHe's open to **remote, hybrid, or on-site** opportunities across India and internationally. Contact: rohithtnsp@gmail.com`;
    }

    // ── 23. HANDLE ROHIT-SPECIFIC CASUAL QUESTIONS ──
    if (/^(tell me about rohit|about rohit|rohit|pamidi|pamidi rohit)[!?.\s]*$/i.test(q)) {
      return this._formatAbout(profile, this.retrieve(q, 3).map(r => r.doc));
    }

    // ── 24. WHAT/HOW MANY QUESTIONS ──
    if (/how many (projects?|repos?|repositories|apps?|application)/i.test(q)) {
      return `Rohit has built **22+ projects** on GitHub! They span full-stack web development, machine learning/AI, NLP research, and cloud deployment.\n\n**Highlights:**\n- **Tomato** — Full-stack food delivery with Gemini AI chatbot\n- **CineMatch AI** — Movie recommendations (10K+ movies)\n- **HeartDisease ML** — 6 ML models, AWS deployed\n- **NeuroCore** — Multi-agent smart city platform\n\nSay *"list all projects"* for the complete portfolio!`;
    }

    if (/how many (leetcode|problems?|questions?|solved)/i.test(q)) {
      return `Rohit has solved **600+ problems** on [LeetCode](https://leetcode.com/u/rohithtnsp/)! 🏆\n\nTopics mastered: Arrays, Trees, Graphs, Dynamic Programming, Backtracking, Binary Search, Heaps, Greedy algorithms, and more.\n\nThis demonstrates strong algorithmic problem-solving skills that power his efficient backend code.`;
    }

    // ── 24. GENERIC "WHAT IS/WHAT'S" without tech context ──  
    if (/^what('?s| is) (your|the) (purpose|goal|point|use|function)/i.test(q)) {
      return `My purpose is to help you learn about **Pamidi Rohit** — a B.Tech AI student and full-stack developer.\n\nI have detailed knowledge of his **22+ projects**, **technical skills** (React, Node.js, Python, PyTorch, Docker), **achievements** (600+ LeetCode, ISRO hackathon), and **education** at IIIT Sri City.\n\nJust ask naturally and I'll find the answer!`;
    }

    // ── 27. TECH DEFINITION QUESTIONS — contextualized to Rohit ──
    const techMatch = q.match(/^(?:what is|what's|define|explain)\s+(machine learning|ml|artificial intelligence|ai|docker|react|node\.?js|python|rag|api|rest|graphql|mongodb|postgresql|next\.?js|typescript|flask|nestjs|mlops|multi.?agent|langchain|langgraph|pytorch|scikit.?learn|xgboost|redis|aws|firebase|socket\.?io|jwt|oauth|tailwind|vae|nlp|reinforcement learning|deep learning|tensorflow|pandas|numpy)/i);
    if (techMatch) {
      const topic = techMatch[1].toLowerCase();
      return this._generalTechAnswer(topic, profile);
    }

    // Not a conversational query — return null to proceed with TF-IDF retrieval
    return null;
  }

  // ─── General Tech Answer (kept for direct "what is X" questions) ──

  _generalTechAnswer(topic, profile) {
    const answers = {
      'machine learning': `**Machine Learning** is a subset of AI where systems learn patterns from data.\n\n**Rohit's ML work:** HeartDisease ML (6 models, XGBoost, 90%+ accuracy), CineMatch AI (TF-IDF recommendations, 10K+ movies), Maternal Health Risk prediction, Healthcare Disease Prediction, and Text Summarization.`,
      'ml': `**Machine Learning** — Rohit's ML toolkit: Python, scikit-learn, XGBoost, PyTorch, TF-IDF, pandas, numpy. Projects: HeartDisease ML, CineMatch AI, Maternal Health, RL Traffic Control.`,
      'artificial intelligence': `**AI** — Rohit studies B.Tech in AI & Data Science at IIIT Sri City. He's built Knowledge Graph RAG, AI Search Chat PDF, CineMatch AI, Human-Aligned AI research, and NeuroCore multi-agent smart city.`,
      'ai': `**AI** — Rohit is pursuing B.Tech in AI & Data Science with 22+ projects spanning classical ML, deep learning, NLP, computer vision, and multi-agent systems.`,
      'docker': `**Docker** — Rohit containerizes ALL production projects. Used in: Tomato (multi-container), HeartDisease (AWS EC2), CineMatch, NestJS API, WanderLust, BeyondChats.`,
      'react': `**React** — Used in: Tomato Food Delivery (3 panels with Vite), HeartDisease ML (Chart.js), Brinavv task management. Also Next.js (React framework) for CineMatch AI and AI Search Chat PDF.`,
      'node.js': `**Node.js** — Rohit's primary backend runtime. Used in: Tomato (Socket.io, Razorpay), WanderLust (9 routers, Stripe), BeyondChats, QuickCart, Brinavv, NeuroCore.`,
      'nodejs': `**Node.js** — Primary backend for Tomato, WanderLust, QuickCart, BeyondChats, Brinavv, and NeuroCore. Also NestJS for enterprise TypeScript APIs.`,
      'python': `**Python** — Rohit's ML/AI language: HeartDisease, CineMatch, Maternal Health, RL Traffic Control, Telugu VAE, Text Summarization, Knowledge Graph RAG, Human-Aligned AI. Backends: Flask, FastAPI.`,
      'rag': `**RAG (Retrieval-Augmented Generation)** — Rohit built KGARS (Knowledge Graph RAG with graph traversal) AND this chatbot (custom TF-IDF + cosine similarity RAG engine, no external APIs).`,
      'api': `**APIs** — Rohit builds RESTful APIs with Express.js, NestJS (51 unit tests), Flask (ML serving), and FastAPI (SSE streaming). Also GraphQL experience.`,
      'rest': `**REST** — Implemented across all backend projects with proper status codes, JWT auth, validation, error handling, rate limiting, and CORS.`,
      'mongodb': `**MongoDB** — Used in: Tomato (6 models), WanderLust (Atlas cloud), CineMatch (watchlists), BeyondChats, QuickCart, Brinavv. Mongoose ODM, MongoStore sessions.`,
      'postgresql': `**PostgreSQL** — Used with Supabase in NestJS Production API. TypeORM for entities/migrations, 51 unit tests, enterprise-grade CRUD with auth.`,
      'next.js': `**Next.js** — CineMatch AI (Next.js 14, App Router, SSR) and AI Search Chat PDF (Next.js 16, SSE streaming, FastAPI backend).`,
      'nextjs': `**Next.js** — Rohit uses Next.js 14 (CineMatch AI) and Next.js 16 (AI Search Chat PDF Viewer) for full-stack production apps.`,
      'typescript': `**TypeScript** — Used in NestJS Production API (enterprise-grade, 51 tests), CineMatch AI (Next.js), AI Search Chat PDF. Rohit's preferred language for large-scale apps.`,
      'flask': `**Flask** — Serves ML models: HeartDisease (6 models), CineMatch AI (Flask-Caching, Flask-Limiter, Redis), Maternal Health prediction APIs.`,
      'nestjs': `**NestJS** — Production API with PostgreSQL/Supabase, 51 unit tests, TypeORM, JWT guards, validation pipes, global exception filters, Helmet, enterprise patterns.`,
      'mlops': `**MLOps** — Docker containerization, CI/CD pipelines, pickle model versioning, AWS EC2 deployment (40% latency optimization), GridSearchCV experiment tracking, feature engineering pipelines.`,
      'multi-agent': `**Multi-Agent Systems** — NeuroCore (6 AI agents with Consensus Engine), KGARS (specialized retrieval agents), Tomato (microservices with Socket.io).`,
      'multiagent': `**Multi-Agent Systems** — NeuroCore smart city (6 agents), Knowledge Graph RAG (agent pipeline), Tomato (autonomous microservices), Human-Aligned AI (multi-agent alignment).`,
      'graphql': `**GraphQL** — Experience alongside REST APIs for flexible data fetching with Node.js and NestJS.`,
      'langgraph': `**LangGraph** — Used for building stateful agentic workflows. Applied in NeuroCore (6 AI agents, Consensus Engine) and Knowledge Graph RAG.`,
      'langchain': `**LangChain** — Used for recommendation pipelines and agentic AI workflows in multiple projects.`,
      'pytorch': `**PyTorch** — Used in BAH25 Lunar Hazard Detection (physics-informed neural network, attention-based sensor fusion) and Telugu VAE BTP (Variational Autoencoder research).`,
      'scikit-learn': `**scikit-learn** — Core ML library used in HeartDisease (6 models, GridSearchCV), CineMatch (TF-IDF), Maternal Health (RandomForest), Healthcare Disease Prediction.`,
      'xgboost': `**XGBoost** — Used in HeartDisease ML for achieving superior accuracy among 6 competing models. Gradient boosting champion.`,
      'redis': `**Redis** — Caching in Tomato (81% API speedup, 800ms→150ms) and CineMatch (Flask-Caching with Redis backend, 300-600s TTL).`,
      'aws': `**AWS** — HeartDisease ML deployed on EC2 with 40% latency reduction. Docker containerization for cloud deployment.`,
      'firebase': `**Firebase** — Used in NeuroCore smart city platform (Firestore, Studio, Hosting) for real-time data and deployment.`,
      'socket.io': `**Socket.io** — Real-time WebSockets in Tomato food delivery for order status updates, delivery tracking, and admin management.`,
      'jwt': `**JWT (JSON Web Tokens)** — Authentication in Tomato (Express + Passport), NestJS API (JWT guards + strategy), and other backend projects.`,
      'oauth': `**OAuth** — Google and Facebook OAuth via Passport.js in Tomato food delivery platform.`,
      'tailwind': `**Tailwind CSS** — Used in CineMatch AI and AI Search Chat PDF Viewer for utility-first responsive styling.`,
      'vae': `**VAE (Variational Autoencoder)** — Telugu VAE BTP research project. Trains VAE on Telugu text for latent representations, text generation, and data augmentation.`,
      'nlp': `**NLP** — Text Summarization (extractive + abstractive), Telugu VAE (Dravidian language), TF-IDF in CineMatch and this RAG chatbot, Knowledge Graph RAG.`,
      'reinforcement learning': `**Reinforcement Learning** — RL Traffic Control using DQN/Q-Learning for adaptive traffic signal optimization. State: queue lengths + phase timing. Reward: minimize waiting time.`,
      'deep learning': `**Deep Learning** — PyTorch for physics-informed neural networks (BAH25), VAE for Telugu NLP, MLP in HeartDisease, transformer-based summarization.`,
      'tensorflow': `**TensorFlow** — Used alongside PyTorch in deep learning projects including Telugu VAE BTP research.`,
      'pandas': `**pandas** — Core data processing in CineMatch AI (10K+ movie DataFrame), HeartDisease (feature engineering), and all ML projects.`,
      'numpy': `**NumPy** — Cosine similarity matrix computation in CineMatch AI, numerical operations across all ML projects.`,
    };
    return answers[topic] || `**${topic}** is a technology in the software engineering domain.\n\nRohit has **22+ projects** spanning many technologies. Ask specifically about his experience with ${topic}, or try: *"What are Rohit's skills?"*`;
  }

  // ─── Handle "Can Rohit do X?" style questions ─────────────────

  _handleAbilityQuestion(query, profile) {
    const q = query.toLowerCase();
    // Extract the skill/technology being asked about
    const allSkills = [
      ...profile.skills.languages,
      ...profile.skills.frontend,
      ...profile.skills.backend,
      ...profile.skills.databases,
      ...profile.skills.mlai,
      ...profile.skills.tools
    ].map(s => s.toLowerCase());

    // Sort by length descending so "c++" matches before "c"
    allSkills.sort((a, b) => b.length - a.length);
    const skillMatch = allSkills.find(skill => {
      const escaped = skill.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      return new RegExp('\\b' + escaped + '\\b', 'i').test(q);
    });
    if (skillMatch) {
      // Find relevant KB entries
      const results = this.retrieve(skillMatch, 3);
      const context = results.length > 0 ? results[0].doc.content.slice(0, 200) : '';
      return `**Yes!** Rohit is skilled in **${skillMatch}**. ✅\n\n${context}...\n\nWant to know more about his experience with ${skillMatch}? Just ask!`;
    }

    // Generic ability answer
    return `**Rohit is a versatile full-stack developer and ML engineer.** His tech stack includes:\n\n- **Languages**: ${profile.skills.languages.join(', ')}\n- **Frontend**: ${profile.skills.frontend.slice(0, 4).join(', ')}\n- **Backend**: ${profile.skills.backend.join(', ')}\n- **ML/AI**: ${profile.skills.mlai.slice(0, 6).join(', ')}\n- **DevOps**: ${profile.skills.tools.slice(0, 4).join(', ')}\n\nAsk about a specific technology and I'll tell you exactly how Rohit has used it!`;
  }

  // ─── Handle Short Queries (single word / 2-3 words) ───────────

  _handleShortQuery(q, profile) {
    const shortMap = {
      'name': `His name is **Pamidi Rohit**.`,
      'email': `Rohit's email: **rohithtnsp@gmail.com**`,
      'phone': `Rohit's phone: **+91 9398026237**`,
      'whatsapp': `WhatsApp Rohit: **+91 9398026237** — [Open WhatsApp](https://wa.me/919398026237)`,
      'linkedin': `Rohit's LinkedIn: [rohit-pamidi](${profile.linkedin})`,
      'github': `Rohit's GitHub: [PAMIDIROHIT](${profile.github}) — 22+ repositories`,
      'leetcode': `Rohit's LeetCode: [rohithtnsp](${profile.leetcode}) — **600+ problems** solved`,
      'cgpa': `Rohit's CGPA: **${profile.cgpa}** at IIIT Sri City`,
      'gpa': `Rohit's CGPA: **${profile.cgpa}** at IIIT Sri City`,
      'college': `Rohit studies at **IIIT Sri City** — Indian Institute of Information Technology, Sri City, Andhra Pradesh. B.Tech in AI & Data Science (2023-2027).`,
      'degree': `**B.Tech in Artificial Intelligence and Data Science** at IIIT Sri City (Aug 2023 - May 2027). CGPA: ${profile.cgpa}.`,
      'resume': `Rohit's portfolio is right here! For a formal resume, contact him at **rohithtnsp@gmail.com** or [LinkedIn](${profile.linkedin}).`,
      'cv': `Contact Rohit for a CV at **rohithtnsp@gmail.com** or [LinkedIn](${profile.linkedin}).`,
      'portfolio': `You're looking at it! 🎉 This portfolio includes Rohit's 22+ projects, skills, education, achievements, and this custom-built RAG chatbot. Explore [his GitHub](${profile.github}) for code.`,
      'projects': null, // Let the main handler deal with these
      'skills': null,
      'experience': null,
    };

    const stripped = q.replace(/[^a-z\s]/g, '').trim();
    if (stripped in shortMap) {
      return shortMap[stripped];
    }
    return null;
  }

  // ─── Format Hire Response ─────────────────────────────────────

  _formatHire(profile) {
    return `## Why You Should Hire Pamidi Rohit 🚀\n\n**1. Full-Stack Versatility**\nFrontend (React, Next.js) → Backend (Node.js, NestJS, Flask, FastAPI) → Databases (MongoDB, PostgreSQL, Redis) → ML/AI (scikit-learn, PyTorch) → DevOps (Docker, AWS) — he can own the **entire stack**.\n\n**2. Proven Builder (22+ Projects)**\n- Production food delivery platform with real-time WebSockets & Gemini AI\n- Movie recommendation engine (10K+ movies, 95% accuracy)\n- Enterprise NestJS API with **51 unit tests**\n- Medical ML system deployed on AWS EC2\n\n**3. Strong Foundations**\n- **600+ LeetCode problems** solved — deep algorithmic thinking\n- CGPA **8.23** at IIIT Sri City (institute of national importance)\n\n**4. Cutting-Edge AI**\n- Physics-informed neural networks for ISRO hackathon\n- Multi-agent AI platform with 6 specialized agents\n- Knowledge Graph RAG system\n- LangGraph for agentic workflows\n\n**5. Production Mindset**\n- Dockerizes everything, writes tests, optimizes performance (81% API speedup)\n- Handles payment integration, OAuth, real-time WebSockets\n\n**Contact:** [${profile.email}](mailto:${profile.email}) | [LinkedIn](${profile.linkedin}) | [GitHub](${profile.github})`;
  }

  // ─── Format Fun Facts ────────────────────────────────────────

  _formatFunFact(profile) {
    const facts = [
      `🧠 **Did you know?** Rohit's physics-informed neural network understands **lunar gravity (1.62 m/s²)** and regolith cohesion properties. He built this for ISRO's Bharatiya Antariksh Hackathon 2025!`,
      `⚡ **Speed demon!** Rohit reduced API response time from **800ms to 150ms** (81% improvement) using Redis caching in his Tomato food delivery app.`,
      `🎬 **Movie buff meets ML!** Rohit's CineMatch AI recommends from **10,000+ movies** using TF-IDF and cosine similarity with 95% accuracy.`,
      `🏆 **600+ LeetCode problems!** That's more than most professional developers solve in their entire career. Rohit has mastered DP, Graphs, Trees, Backtracking, and more.`,
      `🤖 **This chatbot you're using** was built FROM SCRATCH by Rohit using TF-IDF vectorization and cosine similarity — no ChatGPT, no external AI APIs!`,
      `🏙️ **City of AI agents!** Rohit's NeuroCore platform has 6 AI agents that literally **vote and negotiate** to make city-wide decisions.`,
      `📝 **Telugu for AI!** Rohit is researching Variational Autoencoders for Telugu — a language spoken by 75+ million people but severely underrepresented in NLP.`,
      `🐳 **Docker everything!** Rohit containerizes ALL his production projects with Docker. It's not an afterthought — it's standard practice.`,
      `📊 **Model showdown!** In his HeartDisease project, Rohit trained 6 ML models simultaneously (Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost, Neural Network) and let them compete.`,
      `🛰️ **Space tech!** Rohit built a system that analyzes actual Chandrayaan satellite imagery (TMC, DTM, OHRC) to detect lunar hazards.`
    ];
    return facts[Math.floor(Math.random() * facts.length)] + `\n\nWant to hear more? Just say *"another fun fact"* or ask about any specific topic!`;
  }

  // ─── Smart Fallback (instead of "I don't know") ──────────────

  _smartFallback(query, profile) {
    // Try fuzzy matching against knowledge base keywords
    const queryWords = query.toLowerCase().split(/\s+/).filter(w => w.length > 2);
    let bestMatch = null;
    let bestScore = 0;

    for (const doc of this.documents) {
      let score = 0;
      const docText = (doc.title + ' ' + doc.keywords.join(' ')).toLowerCase();
      for (const word of queryWords) {
        if (docText.includes(word)) score += 2;
        // Partial match (3+ chars)
        for (const kw of doc.keywords) {
          if (kw.toLowerCase().includes(word) || word.includes(kw.toLowerCase())) score += 1;
        }
      }
      if (score > bestScore) {
        bestScore = score;
        bestMatch = doc;
      }
    }

    if (bestMatch && bestScore >= 2) {
      const snippet = bestMatch.content.slice(0, 300);
      return `I think you might be asking about **${bestMatch.title}**:\n\n${snippet}...\n\nIs this what you were looking for? If not, try rephrasing your question!`;
    }

    // Final fallback — still helpful
    return `I couldn't find a specific answer for *"${query}"*, but here's a quick overview of what I know about **Pamidi Rohit**:\n\n- 🧑 **B.Tech AI & Data Science** at IIIT Sri City (CGPA: ${profile.cgpa})\n- 💻 **22+ projects** — Tomato Food App, CineMatch AI, HeartDisease ML, NeuroCore, and more\n- 🛠 **Full-stack** — React, Node.js, Next.js, Python, Flask, NestJS, Docker\n- 🤖 **ML/AI** — scikit-learn, PyTorch, LangChain, RAG, Multi-Agent Systems\n- 🏆 **600+ LeetCode** problems solved\n- 📬 **Contact**: rohithtnsp@gmail.com\n\nTry asking something specific like:\n- *"Tell me about the Tomato app"*\n- *"What ML skills does Rohit have?"*\n- *"How many projects?"*`;
  }

  getChunkDiagnostics(limit = 10) {
    const ranked = [...this.chunks].sort((a, b) => b.quality - a.quality);
    const avgQuality = ranked.reduce((sum, chunk) => sum + chunk.quality, 0) / Math.max(1, ranked.length);
    const summarize = chunk => ({
      id: chunk.id,
      title: chunk.title,
      type: chunk.type,
      quality: chunk.quality,
      preview: this._trimWords(chunk.rawText || chunk.text, 20),
    });

    return {
      totalChunks: ranked.length,
      avgQuality: Number(avgQuality.toFixed(3)),
      top: ranked.slice(0, limit).map(summarize),
      bottom: ranked.slice(-limit).reverse().map(summarize),
    };
  }

  _collectExpandedEvidence(result, radius = 1) {
    const collected = [];
    const seen = new Set();
    const addChunk = chunk => {
      if (!chunk || seen.has(chunk.id)) return;
      seen.add(chunk.id);
      collected.push(chunk);
    };

    addChunk(result.chunk);
    (result.evidence || []).forEach(addChunk);

    const index = this.chunkIndexById.get(result.chunk.id);
    if (typeof index === 'number') {
      for (let offset = -radius; offset <= radius; offset++) {
        if (offset === 0) continue;
        const neighbor = this.chunks[index + offset];
        if (neighbor && neighbor.docId === result.doc.id && neighbor.type === 'passage') {
          addChunk(neighbor);
        }
      }
    }

    return collected;
  }

  _extractGroundedFacts(query, results, maxFacts = 5) {
    const queryTokens = new Set(this._tokenize(query));
    const candidates = [];

    results.slice(0, 4).forEach((result, resultIndex) => {
      const expanded = this._collectExpandedEvidence(result, 1);
      const naturalChunks = expanded.filter(chunk => chunk.type === 'passage' || chunk.type === 'summary');
      const chunksForAnswer = naturalChunks.length > 0
        ? naturalChunks
        : expanded.filter(chunk => chunk.type !== 'dialogue');

      chunksForAnswer.forEach(chunk => {
        const sourceText = (chunk.rawText || chunk.text || '').replace(/\s+/g, ' ').trim();
        this._splitSentences(sourceText).slice(0, 6).forEach((sentence, sentenceIndex) => {
          const cleaned = sentence.trim();
          if (this._wordCount(cleaned) < 7) return;
          const sentenceTokens = this._tokenize(cleaned);
          const overlap = sentenceTokens.reduce((count, token) => count + (queryTokens.has(token) ? 1 : 0), 0);
          const keywordOverlap = (result.doc.keywords || []).reduce((count, keyword) => (
            cleaned.toLowerCase().includes(String(keyword).toLowerCase()) ? count + 1 : count
          ), 0);

          candidates.push({
            sentence: cleaned,
            score: (result.score * 0.55) + (overlap * 0.18) + (keywordOverlap * 0.04) + (resultIndex === 0 ? 0.2 : 0) - (sentenceIndex * 0.015),
          });
        });
      });
    });

    const seen = new Set();
    return candidates
      .sort((a, b) => b.score - a.score)
      .filter(item => {
        const key = item.sentence.toLowerCase().replace(/[^a-z0-9\s]/g, '').slice(0, 120);
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      })
      .slice(0, maxFacts)
      .map(item => item.sentence.replace(/\s+/g, ' ').trim());
  }

  _isFocusedQueryForDoc(query, doc) {
    const queryTokens = new Set(this._tokenize(query));
    const titleOverlap = this._tokenize(doc.title).reduce((count, token) => count + (queryTokens.has(token) ? 1 : 0), 0);
    const keywordOverlap = (doc.keywords || []).slice(0, 10).reduce((count, keyword) => (
      query.toLowerCase().includes(String(keyword).toLowerCase()) ? count + 1 : count
    ), 0);
    return titleOverlap >= 2 || keywordOverlap >= 2;
  }

  _formatRichAnswer(query, results, options = {}) {
    const top = results[0];
    if (!top) return this._unknownResponse(query);

    const focusedResults = this._isFocusedQueryForDoc(query, top.doc)
      ? [top]
      : results.filter(item => item.score >= top.score * 0.55).slice(0, 4);

    const facts = this._extractGroundedFacts(query, focusedResults, options.maxFacts || 5);
    const lead = facts[0] || this._trimWords(top.chunk.rawText || top.doc.content, 48);
    const bullets = facts.slice(1);
    const related = results.slice(1, 4).map(item => item.doc.title);
    const techFocus = (top.doc.keywords || []).slice(0, 8).join(', ');

    let response = `## ${top.doc.title}\n\n${lead}`;

    if (bullets.length > 0) {
      response += `\n\n### Key Details\n- ${bullets.join('\n- ')}`;
    }

    if (techFocus && options.includeTechFocus !== false) {
      response += `\n\n### Tech / Focus Areas\n${techFocus}`;
    }

    if (related.length > 0) {
      response += `\n\n### Related Context\n- ${related.join('\n- ')}`;
    }

    return response;
  }

  _formatSources(results, maxSources = 5) {
    if (!results || results.length === 0) return '';

    const unique = [];
    const seen = new Set();

    for (const item of results) {
      if (!item || !item.doc || seen.has(item.doc.id)) continue;
      seen.add(item.doc.id);
      unique.push(item);
      if (unique.length >= maxSources) break;
    }

    if (unique.length === 0) return '';

    const lines = unique.map((item, index) => {
      const label = item.doc.title;
      const link = item.source || this._sourceForDoc(item.doc.id);
      const rel = item.score ? ` (relevance: ${item.score.toFixed(2)})` : '';
      if (link) return `${index + 1}. [${label}](${link})${rel}`;
      return `${index + 1}. ${label}${rel}`;
    });

    return `\n\n### Sources\n${lines.join('\n')}`;
  }

  _attachSources(answer, results) {
    if (!answer) return answer;
    if (/### Sources/i.test(answer)) return answer;
    return `${answer}${this._formatSources(results)}`;
  }

  // ─── Formatters ───────────────────────────────────────────────────────────

  _formatAbout(profile, docs) {
    return `## Pamidi Rohit\n\n**${profile.name}** is a ${profile.degree} student at **${profile.college}** with a **CGPA of ${profile.cgpa}**. He is a full-stack developer and ML engineer who has built **22+ production-grade projects** across web engineering, AI systems, RAG, and cloud deployment.\n\n### Why He Stands Out\n- Ships end-to-end systems across React, Next.js, Node.js, NestJS, Flask, FastAPI, MongoDB, PostgreSQL, Redis, Docker, and AWS\n- Strong ML depth across scikit-learn, XGBoost, PyTorch, NLP, RAG, and multi-agent systems\n- Solved **600+ LeetCode problems**, showing strong algorithmic foundations\n- Built standout projects like the **Lunar Hazard Detection System** for ISRO's BAH25 and **NeuroCore**, a 6-agent smart-city AI platform\n\nAsk me to *tell me more* if you want a recruiter-style deep dive.`;
  }

  _formatContact(profile) {
    return `You can reach Rohit at:\n\n- **Email**: [${profile.email}](mailto:${profile.email})\n- **WhatsApp**: [+91 ${profile.whatsapp}](https://wa.me/91${profile.whatsapp})\n- **LinkedIn**: [rohit-pamidi](${profile.linkedin})\n- **GitHub**: [PAMIDIROHIT](${profile.github})\n\nHe's open to internships, full-time roles, freelance projects, and open-source collaboration.`;
  }

  _formatEducation(profile, docs) {
    return `## Education\n\nRohit is pursuing **B.Tech in Artificial Intelligence and Data Science** at **IIIT Sri City** from **Aug 2023 to May 2027** with a **CGPA of ${profile.cgpa}**. Before that, he completed Intermediate at **Sri Chaitanya Junior College** in the MPC stream with **93.7%**.\n\n### Academic Signal\n- Formal specialization in AI, Data Science, and software engineering\n- Strong mathematical base from MPC\n- Academic performance that supports his hands-on engineering portfolio\n\nAsk me to *go deeper* if you want more on coursework and technical direction.`;
  }

  _formatAchievements(docs) {
    return `## Achievements\n\n- **600+ LeetCode problems solved** across DSA, dynamic programming, graphs, trees, binary search, and backtracking\n- **ISRO BAH25 participant** through Team Lunar Pioneers, building a lunar hazard detection system with physics-informed neural networks and Chandrayaan imagery\n- **NeuroCore builder** through Team KernelCrew, creating a multi-agent AI smart-city platform with 6 specialized agents\n- Strong academic performance with **CGPA 8.23** at IIIT Sri City and **93.7%** in Intermediate\n\nThese are strong signals of both engineering depth and execution quality. Ask me to *tell me more* about any one of them.`;
  }

  _formatHackathon(docs, queryLower) {
    const isLunar = /lunar|bah|moon|chandrayaan|isro|space|pioneer|ps.?11/.test(queryLower);
    const isNeuro = /neurocore|agentic|smart.?city|kernelcrew|kernel.?crew|agent/.test(queryLower);

    if (isNeuro) {
      return `## NeuroCore — Multi-Agent Smart City Platform\n\n**NeuroCore** is an agentic AI platform built by Team **KernelCrew** to tackle city-data overload with coordinated AI decision making. Instead of a single assistant, it uses **6 specialized agents** for traffic, safety, health, environment, emergency response, and citizen sentiment.\n\n### Technical Highlights\n- Agents collaborate through a **Consensus Engine** so decisions are cross-domain, not isolated\n- Built with **Gemini 1.5 Pro**, **Vertex AI**, **Google Maps API**, **React + TypeScript**, **Node.js**, **MongoDB**, and **Firebase**\n- Includes advanced ideas like **Neural Event Mesh**, **City Oracle** what-if simulations, and **privacy-preserving Edge AI**\n\n### Why It Matters\nThis is the kind of project that shows Rohit can think beyond CRUD apps and design system-level AI architecture with coordination, reasoning, and product thinking.\n\nAsk me to *go deeper* for the full architecture.`;
    }
    if (isLunar) {
      return `## Lunar Hazard Detection System\n\nBuilt for **ISRO's Bharatiya Antariksh Hackathon 2025 (PS-11)** by Team **Lunar Pioneers**, this project detects landslides and boulders on the Moon using **Chandrayaan imagery**. Rohit contributed as **Team Member-3** from IIIT Sri City.\n\n### Technical Highlights\n- Fuses data from **TMC**, **DTM**, and **OHRC** instead of relying on a single sensor stream\n- Uses a **Physics-Informed Neural Network** that encodes lunar gravity and regolith behavior into the learning process\n- Adds **attention-based sensor fusion** for pixel-level hazard detection and risk classification\n- Built with **PyTorch**, **OpenCV**, **Rasterio**, **React.js**, **TypeScript**, and **MongoDB**\n\n### Why It Stands Out\nThis is a strong recruiter signal because it combines computer vision, scientific modeling, multimodal learning, and a real national-level problem statement from ISRO.\n\nSay *tell me more* for full technical details.`;
    }
    // general hackathon
    return `Rohit has participated in major hackathons:\n\n1. **BAH25 (ISRO)** -- Team Lunar Pioneers built a Lunar Hazard Detection System using physics-informed neural networks on Chandrayaan data (PS-11)\n2. **Agentic AI Competition** -- Team KernelCrew built NeuroCore, a multi-agent smart city platform with 6 AI agents\n\nAsk about either one specifically for more details.`;
  }

  _formatProjects(docs, queryLower, results = []) {
    const projectDocs = docs.filter(d => d.category === 'project');
    const projectResults = results.filter(item => item.doc.category === 'project');
    if (projectDocs.length === 0) return this._formatGeneral(docs, queryLower, results);

    const isOverview = /all|list|overview|projects/.test(queryLower);
    if (isOverview) {
      return `## Project Portfolio\n\nRohit has built **22+ projects** across full-stack engineering, ML/AI, RAG, cloud deployment, and agentic systems.\n\n### Standout Projects\n- **Tomato** — production food delivery platform with Gemini AI chatbot, Redis caching, Socket.io, payments, and multi-panel architecture\n- **CineMatch AI** — movie recommendation system for **10K+ movies** with TF-IDF/cosine similarity and a Next.js frontend\n- **HeartDisease ML** — applied ML product with **6 competing models**, XGBoost, AWS deployment, and strong evaluation discipline\n- **NestJS API** — enterprise-grade TypeScript backend with PostgreSQL/Supabase and **51 unit tests**\n- **AI Search Chat PDF** — Perplexity-style streaming AI UX with SSE and cited PDF workflows\n- **Lunar Hazard Detection** — ISRO BAH25 project using physics-informed neural networks and Chandrayaan imagery\n- **NeuroCore** — agentic AI smart-city platform with **6 specialized agents** and a consensus engine\n\nAsk about any one project and I’ll give you a deeper technical breakdown with grounded sources.`;
    }

    return this._formatRichAnswer(queryLower, projectResults.length > 0 ? projectResults : results, {
      maxFacts: 5,
      includeTechFocus: true,
    });
  }

  _formatSkills(profile, docs, queryLower = '') {
    const s = profile.skills;

    const webSkills = [
      ...s.frontend,
      ...s.backend.filter(skill => /node|express|nestjs|graphql|fastapi|flask/i.test(skill)),
      'REST APIs',
      'WebSocket (Socket.io)',
      'Authentication (JWT/OAuth)'
    ];

    const mlDlSkills = [
      'scikit-learn',
      'XGBoost',
      'PyTorch',
      'TensorFlow',
      'Feature Engineering',
      'Model Evaluation (Accuracy/F1/ROC-AUC)',
      'Cross-Validation',
      'Computer Vision (OpenCV)',
      'Reinforcement Learning (DQN/Q-Learning)'
    ];

    const genAiSkills = [
      'RAG (Retrieval-Augmented Generation)',
      'Knowledge-Graph RAG',
      'LangChain',
      'LangGraph',
      'Prompt Engineering',
      'Vector Search and Similarity Retrieval',
      'Multi-Agent Systems',
      'Gemini API Integrations',
      'Streaming AI UX (SSE)'
    ];

    const wantsWeb = /web|frontend|backend|full.?stack|react|next|node|express|api|javascript|typescript|html|css/.test(queryLower);
    const wantsMlDl = /ml|machine learning|deep learning|model|pytorch|xgboost|scikit|dqn|reinforcement|cv|vision|training/.test(queryLower);
    const wantsGenAi = /genai|generative|llm|rag|agentic|langchain|langgraph|prompt|embedding|retrieval/.test(queryLower);

    if (wantsWeb && !wantsMlDl && !wantsGenAi) {
      return `## Web Development Skills\n\n- ${webSkills.join('\n- ')}\n\nRohit can ship production web systems end-to-end: UI, API, auth, real-time features, and deployment.`;
    }

    if (wantsMlDl && !wantsGenAi && !wantsWeb) {
      return `## ML / Deep Learning Skills\n\n- ${mlDlSkills.join('\n- ')}\n\nHe has built multiple applied ML systems, compared competing models, and deployed them with measurable performance gains.`;
    }

    if (wantsGenAi && !wantsMlDl && !wantsWeb) {
      return `## Generative AI Skills\n\n- ${genAiSkills.join('\n- ')}\n\nHe focuses on grounded generation, retrieval quality, and agentic orchestration for production-grade GenAI experiences.`;
    }

    return `## Technical Skills\n\n### Web Development\n- ${webSkills.join('\n- ')}\n\n### ML / Deep Learning\n- ${mlDlSkills.join('\n- ')}\n\n### Generative AI\n- ${genAiSkills.join('\n- ')}\n\nIf you ask specifically for *web*, *ML/DL*, or *GenAI*, I will show only that section.`;
  }

  _formatGeneral(docs, query, results = []) {
    const best = docs[0];
    if (!best) return this._unknownResponse(query);
    return this._formatRichAnswer(query, results.length > 0 ? results : [{ doc: best, chunk: { rawText: best.content }, score: 1, evidence: [] }], {
      maxFacts: 5,
      includeTechFocus: best.category === 'project' || best.category === 'skills',
    });
  }

  // ─── Detailed Follow-Up ─────────────────────────────────────────────────

  _detailedResponse(topic, profile) {
    const docs = topic.docs;
    switch (topic.type) {
      case 'about':
        return `## About Pamidi Rohit\n\n**${profile.name}** is a ${profile.degree} student at **${profile.college}** (CGPA: ${profile.cgpa}).\n\n${docs[0] ? docs[0].content : ''}\n\n### Quick Stats\n- **Degree**: ${profile.degree}\n- **Institute**: ${profile.college}\n- **Graduation**: ${profile.graduation}\n- **LeetCode**: ${profile.leetcodeProblems} problems solved\n- **Projects**: 22+ real-world applications\n- **Email**: ${profile.email}\n\nHe is actively looking for **internships and full-time opportunities** in Software Engineering, ML/AI, and Full-Stack Development.`;

      case 'education':
        return `## Education\n\n### Indian Institute of Information Technology (IIIT), Sri City\n- **Degree**: Bachelor of Technology in Artificial Intelligence and Data Science\n- **Duration**: August 2023 -- May 2027 (Expected)\n- **CGPA**: **${profile.cgpa}**\n- IIIT Sri City is a premier institute of national importance\n\n### Sri Chaitanya Junior College\n- **Stream**: MPC (Mathematics, Physics, Chemistry)\n- **Duration**: June 2021 -- June 2023\n- **Score**: **93.7%** -- Outstanding performance\n\nRohit's academic foundation in engineering and mathematics drives his success in competitive programming and AI research.`;

      case 'achievements':
        return `## Achievements & Accomplishments\n\n### LeetCode -- 600+ Problems Solved\nCovers Data Structures (Arrays, Trees, Graphs, Heaps), Algorithms (DP, Backtracking, Binary Search, Greedy), and System Design fundamentals.\nProfile: [leetcode.com/u/rohithtnsp](https://leetcode.com/u/rohithtnsp/)\n\n### Bharatiya Antariksh Hackathon 2025 (ISRO)\nTeam **Lunar Pioneers** -- PS-11: AI-powered Lunar Hazard Detection System combining TMC, DTM, OHRC Chandrayaan data with physics-informed neural networks. Tech: PyTorch, OpenCV, Rasterio, React.js.\n\n### NeuroCore Agentic AI\nTeam **KernelCrew** -- Multi-agent smart city platform with 6 specialized AI agents using Gemini 1.5 Pro, Vertex AI, Firebase.\n\n### Academic Excellence\n- CGPA: 8.23/10.0 at IIIT Sri City\n- 93.7% in Intermediate education\n\n### Project Portfolio\nBuilt **22+ production-ready applications** spanning full-stack web dev, ML/AI, and cloud deployment.`;

      case 'projects': {
        const isOverview = true;
        return `## Rohit's Projects Portfolio\n\nRohit has built **22+ real-world applications** across different domains:\n\n| Project | Tech Stack | Highlight |\n|---------|-----------|----------|\n| **Tomato Food Delivery** | Node.js, React, MongoDB, Redis, Gemini AI | 3-panel, Socket.io, Razorpay, 81% faster API |\n| **CineMatch AI** | Flask, Next.js 14, TF-IDF, MongoDB | 10K+ movies, 95% accuracy |\n| **HeartDisease ML** | Python, Flask, React, AWS EC2 | 6 models, XGBoost, 90%+ accuracy |\n| **WanderLust** | Node.js, Stripe, Cloudinary, MongoDB | Airbnb-like, 9 routers |\n| **NestJS Production API** | TypeScript, NestJS, PostgreSQL, Supabase | 51 unit tests, enterprise |\n| **AI Search Chat PDF** | Next.js 16, FastAPI, SSE | Perplexity-style streaming |\n| **Lunar Hazard Detection** | PyTorch, OpenCV, Rasterio, React.js | BAH25 ISRO hackathon, physics-informed NN |\n| **NeuroCore** | Gemini 1.5 Pro, Vertex AI, React, Firebase | 6 AI agents, smart city |\n| **Knowledge Graph RAG** | Python | Graph-augmented RAG system |\n| **Maternal Health AI** | Python, scikit-learn | Pregnancy risk classification |\n| **RL Traffic Control** | Python, DQN | Adaptive signal optimization |\n| **Telugu VAE BTP** | Python, Jupyter, VAE | Dravidian NLP research |\n\nView all repos: [github.com/PAMIDIROHIT](https://github.com/PAMIDIROHIT)`;
      }

      case 'skills': {
        const s = profile.skills;
        return `## Technical Skills\n\n### Programming Languages\n${s.languages.map(l => '`' + l + '`').join(' - ')}\n\n### Frontend Development\n${s.frontend.map(l => '`' + l + '`').join(' - ')}\n\n### Backend Development\n${s.backend.map(l => '`' + l + '`').join(' - ')}\n\n### Databases\n${s.databases.map(l => '`' + l + '`').join(' - ')}\n\n### Machine Learning & AI\n${s.mlai.map(l => '`' + l + '`').join(' - ')}\n\n### Cloud & DevOps Tools\n${s.tools.map(l => '`' + l + '`').join(' - ')}\n\nRohit is a **Full-Stack Developer** capable of owning the entire development pipeline.`;
      }

      case 'hackathon':
        return docs.map(d => `## ${d.title}\n\n${d.content}`).join('\n\n---\n\n');

      case 'contact':
        return `## Contact Pamidi Rohit\n\n| Channel | Details |\n|---------|---------|\n| **Email** | [${profile.email}](mailto:${profile.email}) |\n| **WhatsApp** | [+91 ${profile.whatsapp}](https://wa.me/91${profile.whatsapp}) |\n| **LinkedIn** | [rohit-pamidi](${profile.linkedin}) |\n| **GitHub** | [PAMIDIROHIT](${profile.github}) |\n| **LeetCode** | [rohithtnsp](${profile.leetcode}) |\n\nRohit is **open to opportunities** including internships, full-time roles, freelance projects, and open-source collaboration.`;

      case 'hire':
        return this._formatHire(profile);

      case 'funfact':
        return this._formatFunFact(profile);

      default:
        return docs.map(d => `## ${d.title}\n\n${d.content}`).join('\n\n---\n\n');
    }
  }

  _unknownResponse(query) {
    return this._smartFallback(query, PROFILE);
  }

  // ─── Evaluation Harness ──────────────────────────────────────────────────

  /**
   * Creates a comprehensive test suite for evaluating RAG performance
   * @returns {Object} Test suite with queries, expected results, and scoring functions
   */
  createEvalHarness() {
    const testQueries = [
      // Contact queries
      { query: 'How to contact Rohit?', category: 'contact', expectedDocs: ['contact'], difficulty: 'easy' },
      { query: 'What is Rohit email and LinkedIn?', category: 'contact', expectedDocs: ['contact'], difficulty: 'easy' },
      { query: 'Is Rohit available for internships?', category: 'contact', expectedDocs: ['contact'], difficulty: 'medium' },
      
      // About queries
      { query: 'Who is Rohit?', category: 'about', expectedDocs: ['about', 'education'], difficulty: 'easy' },
      { query: 'Tell me about Pamidi Rohit', category: 'about', expectedDocs: ['about', 'education'], difficulty: 'easy' },
      { query: 'What does Rohit study?', category: 'education', expectedDocs: ['education'], difficulty: 'medium' },
      
      // Skills queries
      { query: 'What programming languages does Rohit know?', category: 'skills', expectedDocs: ['programming_languages', 'web_skills'], difficulty: 'medium' },
      { query: 'Show me Rohit web development skills', category: 'skills', expectedDocs: ['web_skills', 'programming_languages'], difficulty: 'medium' },
      { query: 'What are Rohit ML and AI skills?', category: 'skills', expectedDocs: ['ml_dl_skills', 'genai_skills'], difficulty: 'hard' },
      { query: 'Does Rohit know React and Node.js?', category: 'skills', expectedDocs: ['web_skills'], difficulty: 'hard' },
      
      // Project queries
      { query: 'Show me Rohit projects', category: 'project', expectedDocs: ['agentic_smart_city', 'neurocore_ai', 'portfolio_rag'], difficulty: 'medium' },
      { query: 'What is NeuroCore AI?', category: 'project', expectedDocs: ['neurocore_ai'], difficulty: 'easy' },
      { query: 'Tell me about the Smart City project', category: 'project', expectedDocs: ['agentic_smart_city'], difficulty: 'medium' },
      { query: 'Which RAG techniques does Rohit use?', category: 'project', expectedDocs: ['portfolio_rag'], difficulty: 'hard' },
      
      // Achievement queries
      { query: 'What hackathons did Rohit win?', category: 'achievement', expectedDocs: ['bah25_champion', 'bah25_pioneer'], difficulty: 'medium' },
      { query: 'Tell me about BAH25', category: 'achievement', expectedDocs: ['bah25_champion', 'bah25_pioneer'], difficulty: 'easy' },
      { query: 'How many LeetCode problems has Rohit solved?', category: 'achievement', expectedDocs: ['leetcode_450'], difficulty: 'medium' },
      
      // Technical depth queries
      { query: 'How does Rohit implement RAG pipelines?', category: 'project', expectedDocs: ['portfolio_rag'], difficulty: 'hard' },
      { query: 'What is ColBERT reranking?', category: 'project', expectedDocs: ['portfolio_rag'], difficulty: 'hard' },
      { query: 'Explain Rohit agentic AI work', category: 'project', expectedDocs: ['agentic_smart_city', 'neurocore_ai'], difficulty: 'hard' },
      
      // Edge cases
      { query: 'Hello', category: 'conversational', expectedDocs: [], difficulty: 'easy' },
      { query: 'What is the meaning of life?', category: 'unknown', expectedDocs: [], difficulty: 'hard' },
      { query: 'Tell me a random fact about Rohit', category: 'funfact', expectedDocs: ['agentic_smart_city', 'neurocore_ai'], difficulty: 'medium' },
    ];

    return {
      queries: testQueries,
      runEvaluation: (engine = this) => this._runFullEvaluation(engine, testQueries),
      computeMetrics: (results) => this._computeEvalMetrics(results),
      generateReport: (metrics) => this._generateEvalReport(metrics),
    };
  }

  /**
   * Run comprehensive evaluation on all test queries
   */
  _runFullEvaluation(engine, testQueries) {
    const results = [];
    const startTime = Date.now();
    
    testQueries.forEach((testCase, index) => {
      const queryStart = Date.now();
      
      try {
        // Run retrieval
        const retrievedResults = engine.retrieve(testCase.query, 6);
        const response = engine.generateResponse(testCase.query);
        
        // Check if expected docs were retrieved
        const retrievedDocIds = retrievedResults.map(r => r.doc.id);
        const expectedHits = testCase.expectedDocs.filter(docId => 
          retrievedDocIds.includes(docId)
        );
        
        // Calculate metrics
        const recall = testCase.expectedDocs.length > 0 ? 
          expectedHits.length / testCase.expectedDocs.length : 1;
        
        const precision = retrievedResults.length > 0 ?
          expectedHits.length / Math.min(retrievedResults.length, 3) : 0;
        
        // Calculate MRR (Mean Reciprocal Rank)
        let mrr = 0;
        if (testCase.expectedDocs.length > 0) {
          for (const expectedDoc of testCase.expectedDocs) {
            const rank = retrievedDocIds.indexOf(expectedDoc) + 1;
            if (rank > 0) {
              mrr = Math.max(mrr, 1 / rank);
              break;
            }
          }
        }
        
        // Score answer quality
        const answerQuality = this._scoreAnswerQuality(
          response, testCase.query, testCase.category
        );
        
        const queryTime = Date.now() - queryStart;
        
        results.push({
          index,
          query: testCase.query,
          category: testCase.category,
          difficulty: testCase.difficulty,
          expectedDocs: testCase.expectedDocs,
          retrievedDocs: retrievedDocIds.slice(0, 3),
          response,
          expectedHits,
          recall,
          precision,
          mrr,
          answerQuality,
          queryTime,
          passed: recall >= 0.5 && answerQuality >= 0.6,
        });
        
      } catch (error) {
        results.push({
          index,
          query: testCase.query,
          category: testCase.category,
          error: error.message,
          passed: false,
          recall: 0,
          precision: 0,
          mrr: 0,
          answerQuality: 0,
          queryTime: Date.now() - queryStart,
        });
      }
    });
    
    const totalTime = Date.now() - startTime;
    return { results, totalTime, timestamp: new Date().toISOString() };
  }

  /**
   * Score answer quality based on content analysis
   */
  _scoreAnswerQuality(response, query, category) {
    if (!response || typeof response !== 'string') return 0;
    
    let score = 0.5; // Base score
    const responseLower = response.toLowerCase();
    const queryLower = query.toLowerCase();
    
    // Length appropriateness (not too short, not too long)
    const words = response.split(/\s+/).length;
    if (words >= 20 && words <= 150) score += 0.15;
    else if (words >= 10 && words <= 250) score += 0.08;
    
    // Contains relevant keywords for category
    const categoryKeywords = {
      contact: ['email', 'linkedin', 'whatsapp', 'contact', 'reach'],
      about: ['rohit', 'pamidi', 'student', 'iiit', 'ai', 'data science'],
      education: ['iiit', 'bachelor', 'cgpa', 'college', 'degree', 'ai', 'data science'],
      skills: ['programming', 'languages', 'web', 'react', 'node', 'python', 'javascript'],
      project: ['project', 'built', 'developed', 'technologies', 'features'],
      achievement: ['hackathon', 'leetcode', 'problems', 'solved', 'champion', 'winner'],
    };
    
    const keywords = categoryKeywords[category] || [];
    const keywordHits = keywords.filter(kw => responseLower.includes(kw)).length;
    score += Math.min(0.2, keywordHits * 0.04);
    
    // Query relevance
    const queryWords = queryLower.split(/\s+/).filter(w => w.length > 2);
    const relevanceHits = queryWords.filter(w => responseLower.includes(w)).length;
    score += Math.min(0.15, (relevanceHits / Math.max(1, queryWords.length)) * 0.15);
    
    return Math.min(1, Math.max(0, score));
  }

  /**
   * Compute comprehensive evaluation metrics
   */
  _computeEvalMetrics(evalResults) {
    const { results } = evalResults;
    const validResults = results.filter(r => !r.error);
    
    if (validResults.length === 0) {
      return { error: 'No valid results to compute metrics' };
    }
    
    // Overall metrics
    const avgRecall = validResults.reduce((sum, r) => sum + r.recall, 0) / validResults.length;
    const avgPrecision = validResults.reduce((sum, r) => sum + r.precision, 0) / validResults.length;
    const avgMRR = validResults.reduce((sum, r) => sum + r.mrr, 0) / validResults.length;
    const avgAnswerQuality = validResults.reduce((sum, r) => sum + r.answerQuality, 0) / validResults.length;
    const passRate = validResults.filter(r => r.passed).length / validResults.length;
    const avgQueryTime = validResults.reduce((sum, r) => sum + r.queryTime, 0) / validResults.length;
    
    // Metrics by category
    const byCategory = {};
    const categories = ['contact', 'about', 'education', 'skills', 'project', 'achievement'];
    
    categories.forEach(category => {
      const categoryResults = validResults.filter(r => r.category === category);
      if (categoryResults.length > 0) {
        byCategory[category] = {
          count: categoryResults.length,
          recall: categoryResults.reduce((sum, r) => sum + r.recall, 0) / categoryResults.length,
          precision: categoryResults.reduce((sum, r) => sum + r.precision, 0) / categoryResults.length,
          mrr: categoryResults.reduce((sum, r) => sum + r.mrr, 0) / categoryResults.length,
          answerQuality: categoryResults.reduce((sum, r) => sum + r.answerQuality, 0) / categoryResults.length,
          passRate: categoryResults.filter(r => r.passed).length / categoryResults.length,
        };
      }
    });
    
    // Metrics by difficulty
    const byDifficulty = {};
    ['easy', 'medium', 'hard'].forEach(difficulty => {
      const diffResults = validResults.filter(r => r.difficulty === difficulty);
      if (diffResults.length > 0) {
        byDifficulty[difficulty] = {
          count: diffResults.length,
          recall: diffResults.reduce((sum, r) => sum + r.recall, 0) / diffResults.length,
          passRate: diffResults.filter(r => r.passed).length / diffResults.length,
          avgScore: diffResults.reduce((sum, r) => sum + (r.recall + r.answerQuality) / 2, 0) / diffResults.length,
        };
      }
    });
    
    return {
      overall: {
        totalQueries: results.length,
        validQueries: validResults.length,
        avgRecall: Number(avgRecall.toFixed(3)),
        avgPrecision: Number(avgPrecision.toFixed(3)),
        avgMRR: Number(avgMRR.toFixed(3)),
        avgAnswerQuality: Number(avgAnswerQuality.toFixed(3)),
        passRate: Number(passRate.toFixed(3)),
        avgQueryTime: Math.round(avgQueryTime),
      },
      byCategory,
      byDifficulty,
      failedQueries: results.filter(r => !r.passed).map(r => ({
        query: r.query,
        category: r.category,
        issue: r.error || `Low performance: R=${r.recall?.toFixed(2)} Q=${r.answerQuality?.toFixed(2)}`,
      })),
    };
  }

  /**
   * Generate human-readable evaluation report
   */
  _generateEvalReport(metrics) {
    const { overall, byCategory, byDifficulty, failedQueries } = metrics;
    
    let report = `# RAG Engine Evaluation Report\n\n`;
    report += `**Generated:** ${new Date().toLocaleString()}\n\n`;
    
    // Overall performance
    report += `## Overall Performance\n\n`;
    report += `- **Queries Tested:** ${overall.totalQueries} (${overall.validQueries} valid)\n`;
    report += `- **Pass Rate:** ${(overall.passRate * 100).toFixed(1)}%\n`;
    report += `- **Average Recall@3:** ${overall.avgRecall}\n`;
    report += `- **Average Precision@3:** ${overall.avgPrecision}\n`;
    report += `- **Mean Reciprocal Rank:** ${overall.avgMRR}\n`;
    report += `- **Answer Quality:** ${overall.avgAnswerQuality}\n`;
    report += `- **Avg Query Time:** ${overall.avgQueryTime}ms\n\n`;
    
    // Performance by category
    report += `## Performance by Category\n\n`;
    report += `| Category | Queries | Recall@3 | Precision@3 | Answer Quality | Pass Rate |\n`;
    report += `|----------|---------|----------|-------------|----------------|-----------|\n`;
    
    Object.entries(byCategory).forEach(([category, stats]) => {
      report += `| ${category} | ${stats.count} | ${stats.recall.toFixed(2)} | ${stats.precision.toFixed(2)} | ${stats.answerQuality.toFixed(2)} | ${(stats.passRate * 100).toFixed(1)}% |\n`;
    });
    
    // Performance by difficulty
    report += `\n## Performance by Difficulty\n\n`;
    report += `| Difficulty | Queries | Recall@3 | Pass Rate | Avg Score |\n`;
    report += `|------------|---------|----------|-----------|-----------|\n`;
    
    Object.entries(byDifficulty).forEach(([difficulty, stats]) => {
      report += `| ${difficulty} | ${stats.count} | ${stats.recall.toFixed(2)} | ${(stats.passRate * 100).toFixed(1)}% | ${stats.avgScore.toFixed(2)} |\n`;
    });
    
    // Failed queries analysis
    if (failedQueries.length > 0) {
      report += `\n## Failed Queries (${failedQueries.length})\n\n`;
      failedQueries.forEach((failure, index) => {
        report += `${index + 1}. **${failure.query}** (${failure.category})\n   - Issue: ${failure.issue}\n\n`;
      });
    }
    
    return report;
  }

  // ─── Debug Panel ─────────────────────────────────────────────────────────

  /**
   * Get detailed retrieval trace for debugging
   * @param {string} query - The query to debug
   * @returns {Object} Complete retrieval trace with visualizations
   */
  getDebugTrace(query) {
    // Run retrieval and capture detailed trace
    const results = this.retrieve(query, 8);
    const trace = this._lastRetrievalTrace;
    
    if (!trace) {
      return { error: 'No retrieval trace available. Run retrieve() first.' };
    }
    
    return {
      query: {
        original: query,
        normalized: this._normalizeCasualText(query.toLowerCase()),
        rewrites: trace.rewritten.rewrites.map(r => ({
          text: r.text,
          weight: r.weight,
          tokenCount: this._tokenize(r.text).length,
        })),
      },
      retrieval: {
        sparse: {
          method: 'BM25',
          topK: this.config.sparseTopK,
          results: trace.sparseTop.map(item => ({
            chunkId: this.chunks[item.chunkIndex]?.id,
            title: this.chunks[item.chunkIndex]?.title,
            type: this.chunks[item.chunkIndex]?.type,
            score: Number(item.score.toFixed(4)),
            quality: this.chunks[item.chunkIndex]?.quality,
            preview: this.chunks[item.chunkIndex]?.rawText?.slice(0, 100) + '...',
          })),
        },
        dense: {
          method: 'BGE-style Dense Embedding',
          topK: this.config.denseTopK,
          results: trace.denseTop.map(item => ({
            chunkId: this.chunks[item.chunkIndex]?.id,
            title: this.chunks[item.chunkIndex]?.title,
            type: this.chunks[item.chunkIndex]?.type,
            score: Number(item.score.toFixed(4)),
            quality: this.chunks[item.chunkIndex]?.quality,
            preview: this.chunks[item.chunkIndex]?.rawText?.slice(0, 100) + '...',
          })),
        },
        fusion: {
          method: 'Reciprocal Rank Fusion (RRF)',
          rrfK: this.config.rrfK,
          results: trace.fusedTop.map(item => ({
            chunkId: this.chunks[item.chunkIndex]?.id,
            title: this.chunks[item.chunkIndex]?.title,
            rrfScore: Number(item.rrfScore.toFixed(4)),
            sparseRank: item.sparseRank,
            denseRank: item.denseRank,
            sparseScore: Number(item.sparseScore.toFixed(3)),
            denseScore: Number(item.denseScore.toFixed(3)),
          })),
        },
        reranking: {
          method: 'ColBERT Late Interaction',
          results: trace.rerankedTop.map(item => ({
            chunkId: this.chunks[item.chunkIndex]?.id,
            title: this.chunks[item.chunkIndex]?.title,
            finalScore: Number(item.finalScore.toFixed(4)),
            colbertScore: Number(item.colbertScore.toFixed(4)),
            rrfScore: Number(item.rrfScore.toFixed(4)),
            quality: this.chunks[item.chunkIndex]?.quality,
          })),
        },
      },
      finalResults: results.map((result, rank) => ({
        rank: rank + 1,
        docId: result.doc.id,
        title: result.doc.title,
        category: result.doc.category,
        score: Number(result.score.toFixed(4)),
        chunkId: result.chunk.id,
        chunkType: result.chunk.type,
        chunkQuality: result.chunk.quality,
        source: result.source,
        preview: result.chunk.rawText?.slice(0, 150) + '...',
        evidenceCount: result.evidence?.length || 1,
      })),
      performance: {
        totalChunks: this.chunks.length,
        averageChunkQuality: Number((this.chunks.reduce((sum, c) => sum + c.quality, 0) / this.chunks.length).toFixed(3)),
        chunksByType: this._getChunkTypeDistribution(),
        retrievalLatency: this._lastRetrievalTime || 0,
      },
      config: {
        chunkSize: this.config.chunkSize,
        sparseTopK: this.config.sparseTopK,
        denseTopK: this.config.denseTopK,
        fusedTopK: this.config.fusedTopK,
        rerankTopK: this.config.rerankTopK,
        rrfK: this.config.rrfK,
      },
    };
  }

  /**
   * Get visualization data for debug panel UI
   */
  getDebugVisualization(query) {
    const trace = this.getDebugTrace(query);
    if (trace.error) return trace;
    
    return {
      queryFlow: this._buildQueryFlowChart(trace),
      scoreDistribution: this._buildScoreDistribution(trace),
      chunkAnalysis: this._buildChunkAnalysis(trace),
      performanceMetrics: this._buildPerformanceChart(trace),
    };
  }

  _getChunkTypeDistribution() {
    const distribution = {};
    this.chunks.forEach(chunk => {
      distribution[chunk.type] = (distribution[chunk.type] || 0) + 1;
    });
    return distribution;
  }

  _buildQueryFlowChart(trace) {
    return {
      type: 'flowchart',
      stages: [
        { name: 'Original Query', value: trace.query.original, count: 1 },
        { name: 'Query Rewrites', value: `${trace.query.rewrites.length} variants`, count: trace.query.rewrites.length },
        { name: 'Sparse Retrieval', value: `BM25 (top ${trace.retrieval.sparse.results.length})`, count: trace.retrieval.sparse.results.length },
        { name: 'Dense Retrieval', value: `BGE (top ${trace.retrieval.dense.results.length})`, count: trace.retrieval.dense.results.length },
        { name: 'RRF Fusion', value: `Merged (top ${trace.retrieval.fusion.results.length})`, count: trace.retrieval.fusion.results.length },
        { name: 'ColBERT Rerank', value: `Reranked (top ${trace.retrieval.reranking.results.length})`, count: trace.retrieval.reranking.results.length },
        { name: 'Final Results', value: `Documents (${trace.finalResults.length})`, count: trace.finalResults.length },
      ],
    };
  }

  _buildScoreDistribution(trace) {
    return {
      type: 'histogram',
      sparse: trace.retrieval.sparse.results.map(r => r.score),
      dense: trace.retrieval.dense.results.map(r => r.score),
      fused: trace.retrieval.fusion.results.map(r => r.rrfScore),
      final: trace.retrieval.reranking.results.map(r => r.finalScore),
    };
  }

  _buildChunkAnalysis(trace) {
    const chunkTypes = new Set();
    const qualityByType = {};
    
    trace.finalResults.forEach(result => {
      chunkTypes.add(result.chunkType);
      if (!qualityByType[result.chunkType]) {
        qualityByType[result.chunkType] = [];
      }
      qualityByType[result.chunkType].push(result.chunkQuality);
    });
    
    return {
      typeDistribution: Array.from(chunkTypes).map(type => ({
        type,
        count: qualityByType[type].length,
        avgQuality: qualityByType[type].reduce((a, b) => a + b, 0) / qualityByType[type].length,
      })),
    };
  }

  _buildPerformanceChart(trace) {
    return {
      type: 'metrics',
      data: [
        { metric: 'Total Chunks', value: trace.performance.totalChunks },
        { metric: 'Avg Quality', value: trace.performance.averageChunkQuality },
        { metric: 'Sparse Retrieved', value: trace.retrieval.sparse.results.length },
        { metric: 'Dense Retrieved', value: trace.retrieval.dense.results.length },
        { metric: 'Final Docs', value: trace.finalResults.length },
      ],
    };
  }

  /**
   * Generate debug report for a query
   */
  generateDebugReport(query) {
    const trace = this.getDebugTrace(query);
    
    if (trace.error) {
      return `# Debug Report\n\nError: ${trace.error}`;
    }
    
    let report = `# RAG Engine Debug Report\n\n`;
    report += `**Query:** "${trace.query.original}"\n`;
    report += `**Generated:** ${new Date().toLocaleString()}\n\n`;
    
    // Query processing
    report += `## Query Processing\n\n`;
    report += `- **Original:** ${trace.query.original}\n`;
    report += `- **Normalized:** ${trace.query.normalized}\n`;
    report += `- **Rewrites:** ${trace.query.rewrites.length}\n\n`;
    
    trace.query.rewrites.forEach((rewrite, i) => {
      report += `  ${i + 1}. "${rewrite.text}" (weight: ${rewrite.weight}, tokens: ${rewrite.tokenCount})\n`;
    });
    
    // Retrieval stages
    report += `\n## Retrieval Pipeline\n\n`;
    
    // Sparse retrieval
    report += `### BM25 Sparse Retrieval (Top ${trace.retrieval.sparse.results.length})\n\n`;
    trace.retrieval.sparse.results.slice(0, 5).forEach((result, i) => {
      report += `${i + 1}. **${result.title}** (${result.type})\n`;
      report += `   - Score: ${result.score} | Quality: ${result.quality}\n`;
      report += `   - Preview: ${result.preview}\n\n`;
    });
    
    // Dense retrieval
    report += `### BGE Dense Retrieval (Top ${trace.retrieval.dense.results.length})\n\n`;
    trace.retrieval.dense.results.slice(0, 5).forEach((result, i) => {
      report += `${i + 1}. **${result.title}** (${result.type})\n`;
      report += `   - Score: ${result.score} | Quality: ${result.quality}\n`;
      report += `   - Preview: ${result.preview}\n\n`;
    });
    
    // Final results
    report += `### Final Ranked Results (${trace.finalResults.length})\n\n`;
    trace.finalResults.forEach((result, i) => {
      report += `${i + 1}. **${result.title}** (${result.category})\n`;
      report += `   - Final Score: ${result.score} | Chunk Quality: ${result.chunkQuality}\n`;
      report += `   - Chunk Type: ${result.chunkType} | Evidence: ${result.evidenceCount} chunks\n`;
      if (result.source) report += `   - Source: ${result.source}\n`;
      report += `   - Preview: ${result.preview}\n\n`;
    });
    
    // Configuration
    report += `## Configuration\n\n`;
    Object.entries(trace.config).forEach(([key, value]) => {
      report += `- **${key}:** ${value}\n`;
    });
    
    return report;
  }

  // ─── Metadata-Aware Chunk Expansion ──────────────────────────────────────

  /**
   * Expand retrieved chunks with contextual neighbors and metadata
   * @param {Array} results - Retrieved results to expand
   * @param {Object} options - Expansion options
   * @returns {Array} Results with expanded context
   */
  expandChunksWithContext(results, options = {}) {
    const {
      neighborhoodSize = 1,
      includeDocumentContext = true,
      prioritizeQuality = true,
      maxExpansionRatio = 2.0,
      preserveChunkTypes = new Set(['summary', 'qa', 'keyword']),
    } = options;
    
    return results.map(result => {
      const expandedChunks = new Set([result.chunk]);
      const docChunks = this.docChunkMap.get(result.doc.id) || [];
      
      // Add neighborhood chunks
      if (neighborhoodSize > 0) {
        const currentChunkIndex = this.chunkIndexById.get(result.chunk.id);
        if (currentChunkIndex !== undefined) {
          this._addNeighborChunks(
            currentChunkIndex, 
            docChunks, 
            neighborhoodSize,
            expandedChunks,
            preserveChunkTypes
          );
        }
      }
      
      // Add document context chunks
      if (includeDocumentContext) {
        this._addDocumentContextChunks(
          result.doc.id,
          docChunks,
          expandedChunks,
          preserveChunkTypes
        );
      }
      
      // Quality-based expansion
      if (prioritizeQuality) {
        this._addHighQualityChunks(
          docChunks,
          expandedChunks,
          maxExpansionRatio,
          result.chunk.quality
        );
      }
      
      const expandedArray = Array.from(expandedChunks);
      
      // Sort by sentence order and quality
      expandedArray.sort((a, b) => {
        // Preserve original chunk at the top
        if (a.id === result.chunk.id) return -1;
        if (b.id === result.chunk.id) return 1;
        
        // Then by sentence start position
        const aStart = a.sentenceStart || 0;
        const bStart = b.sentenceStart || 0;
        if (aStart !== bStart) return aStart - bStart;
        
        // Then by quality
        return b.quality - a.quality;
      });
      
      return {
        ...result,
        expandedChunks: expandedArray,
        expansionMetadata: {
          originalChunkId: result.chunk.id,
          totalChunks: expandedArray.length,
          expansionRatio: expandedArray.length / 1,
          contextTypes: [...new Set(expandedArray.map(c => c.type))],
          qualityRange: {
            min: Math.min(...expandedArray.map(c => c.quality)),
            max: Math.max(...expandedArray.map(c => c.quality)),
            avg: expandedArray.reduce((sum, c) => sum + c.quality, 0) / expandedArray.length,
          },
        },
      };
    });
  }

  _addNeighborChunks(currentIndex, docChunks, neighborhoodSize, expandedChunks, preserveTypes) {
    for (let offset = -neighborhoodSize; offset <= neighborhoodSize; offset++) {
      if (offset === 0) continue;
      
      const neighborIndex = currentIndex + offset;
      if (docChunks.includes(neighborIndex)) {
        const chunk = this.chunks[neighborIndex];
        if (chunk && this._shouldIncludeChunk(chunk, preserveTypes)) {
          expandedChunks.add(chunk);
        }
      }
    }
  }

  _addDocumentContextChunks(docId, docChunks, expandedChunks, preserveTypes) {
    // Add summary and keyword chunks for better context
    docChunks.forEach(chunkIndex => {
      const chunk = this.chunks[chunkIndex];
      if (chunk && (chunk.type === 'summary' || chunk.type === 'keyword')) {
        if (this._shouldIncludeChunk(chunk, preserveTypes)) {
          expandedChunks.add(chunk);
        }
      }
    });
  }

  _addHighQualityChunks(docChunks, expandedChunks, maxExpansionRatio, baseQuality) {
    const currentCount = expandedChunks.size;
    const maxTotal = Math.floor(currentCount * maxExpansionRatio);
    const availableSlots = maxTotal - currentCount;
    
    if (availableSlots <= 0) return;
    
    // Get high-quality chunks not already included
    const candidates = docChunks
      .map(index => this.chunks[index])
      .filter(chunk => 
        chunk && 
        !expandedChunks.has(chunk) && 
        chunk.quality >= baseQuality * 0.85
      )
      .sort((a, b) => b.quality - a.quality)
      .slice(0, availableSlots);
    
    candidates.forEach(chunk => expandedChunks.add(chunk));
  }

  _shouldIncludeChunk(chunk, preserveTypes) {
    // Don't include dialogue chunks in expansion (they're synthetic)
    if (chunk.type === 'dialogue') return false;
    
    // Include all chunks in preserveTypes
    if (preserveTypes.has(chunk.type)) return true;
    
    // Include high-quality passage chunks
    return chunk.type === 'passage' && chunk.quality >= 0.7;
  }

  /**
   * Generate enhanced response using expanded context
   * @param {string} query - User query
   * @param {Object} profile - User profile
   * @param {Object} expansionOptions - Chunk expansion options
   * @returns {string} Enhanced response with expanded context
   */
  generateEnhancedResponse(query, profile, expansionOptions = {}) {
    // Get base retrieval results
    const results = this.retrieve(query, 4);
    
    if (results.length === 0) {
      return this.generateResponse(query, profile);
    }
    
    // Expand chunks with context
    const expandedResults = this.expandChunksWithContext(results, expansionOptions);
    
    // Enhanced fact extraction with expanded context
    const facts = this._extractEnhancedFacts(query, expandedResults);
    
    // Generate response with expanded context
    return this._synthesizeEnhancedAnswer(query, facts, expandedResults, profile);
  }

  _extractEnhancedFacts(query, expandedResults) {
    const facts = [];
    const seen = new Set();
    
    expandedResults.forEach((result, resultIndex) => {
      const { doc, expandedChunks, expansionMetadata } = result;
      
      // Process expanded chunks in order
      expandedChunks.forEach((chunk, chunkIndex) => {
        if (chunk.type === 'dialogue') return; // Skip synthetic
        
        const factId = `${doc.id}:${chunk.type}:${chunkIndex}`;
        if (seen.has(factId)) return;
        seen.add(factId);
        
        const isOriginalChunk = chunk.id === expansionMetadata.originalChunkId;
        const contextWeight = isOriginalChunk ? 1.0 : 0.7;
        const qualityWeight = chunk.quality;
        const positionWeight = Math.max(0.3, 1 - (chunkIndex * 0.15));
        
        const relevanceScore = result.score * contextWeight * qualityWeight * positionWeight;
        
        facts.push({
          text: chunk.rawText,
          title: doc.title,
          category: doc.category,
          chunkType: chunk.type,
          source: chunk.source,
          relevance: relevanceScore,
          quality: chunk.quality,
          isExpanded: !isOriginalChunk,
          expansionSource: isOriginalChunk ? 'original' : 'context',
          docRank: resultIndex + 1,
          chunkRank: chunkIndex + 1,
        });
      });
    });
    
    // Sort by relevance and quality
    return facts
      .sort((a, b) => b.relevance - a.relevance)
      .slice(0, 12); // Limit for response generation
  }

  _synthesizeEnhancedAnswer(query, facts, expandedResults, profile) {
    if (facts.length === 0) {
      return this._smartFallback(query, profile);
    }
    
    // Group facts by document for better organization
    const factsByDoc = new Map();
    facts.forEach(fact => {
      if (!factsByDoc.has(fact.title)) {
        factsByDoc.set(fact.title, []);
      }
      factsByDoc.get(fact.title).push(fact);
    });
    
    // Build enhanced answer sections
    const sections = [];
    let totalExpansionChunks = 0;
    
    Array.from(factsByDoc.entries()).slice(0, 4).forEach(([title, docFacts]) => {
      const primaryFacts = docFacts.filter(f => !f.isExpanded).slice(0, 2);
      const contextFacts = docFacts.filter(f => f.isExpanded).slice(0, 2);
      
      totalExpansionChunks += contextFacts.length;
      
      if (primaryFacts.length > 0) {
        const section = `## ${title}\n\n${this._combineFactTexts(primaryFacts, contextFacts)}`;
        sections.push(section);
      }
    });
    
    let response = sections.length > 1 ? 
      sections.join('\n\n---\n\n') : 
      sections.join('');
    
    // Add enhancement metadata
    if (totalExpansionChunks > 0) {
      response += `\n\n*Enhanced with ${totalExpansionChunks} contextual chunks for comprehensive coverage.*`;
    }
    
    // Attach sources with expansion info
    const sources = Array.from(new Set(facts.map(f => f.source).filter(Boolean)));
    if (sources.length > 0) {
      response += `\n\n**Sources:** ${sources.map(url => `[Link](${url})`).join(' • ')}`;
    }
    
    return response;
  }

  _combineFactTexts(primaryFacts, contextFacts) {
    let text = primaryFacts.map(f => f.text).join(' ');
    
    // Add relevant context facts that provide additional detail
    contextFacts.forEach(contextFact => {
      if (contextFact.chunkType === 'summary') {
        // Prepend summaries for better overview
        text = `${contextFact.text} ${text}`;
      } else if (contextFact.chunkType === 'keyword') {
        // Add keywords for completeness
        text += ` ${contextFact.text}`;
      } else if (contextFact.quality >= 0.8) {
        // Add high-quality passage chunks
        text += ` ${contextFact.text}`;
      }
    });
    
    return text.replace(/\s+/g, ' ').trim();
  }

  /**
   * Get metadata-aware chunk statistics
   */
  getExpansionAnalytics() {
    const docStats = new Map();
    
    // Analyze document structure and chunk distribution
    this.documents.forEach(doc => {
      const chunks = (this.docChunkMap.get(doc.id) || []).map(index => this.chunks[index]);
      
      const chunksByType = {};
      const qualityStats = [];
      const sentenceSpans = [];
      
      chunks.forEach(chunk => {
        chunksByType[chunk.type] = (chunksByType[chunk.type] || 0) + 1;
        qualityStats.push(chunk.quality);
        
        if (chunk.sentenceStart !== undefined && chunk.sentenceEnd !== undefined) {
          sentenceSpans.push(chunk.sentenceEnd - chunk.sentenceStart + 1);
        }
      });
      
      docStats.set(doc.id, {
        title: doc.title,
        category: doc.category,
        totalChunks: chunks.length,
        chunksByType,
        avgQuality: qualityStats.reduce((a, b) => a + b, 0) / qualityStats.length,
        qualityRange: [Math.min(...qualityStats), Math.max(...qualityStats)],
        avgSentenceSpan: sentenceSpans.length > 0 ? 
          sentenceSpans.reduce((a, b) => a + b, 0) / sentenceSpans.length : 0,
        expansionPotential: this._calculateExpansionPotential(chunks),
      });
    });
    
    return {
      documentStats: Array.from(docStats.values()),
      overallStats: this._calculateOverallExpansionStats(docStats),
    };
  }

  _calculateExpansionPotential(chunks) {
    const passageChunks = chunks.filter(c => c.type === 'passage');
    const highQualityChunks = chunks.filter(c => c.quality >= 0.8);
    const neighborhoodDensity = this._calculateNeighborhoodDensity(chunks);
    
    return {
      passageCount: passageChunks.length,
      highQualityCount: highQualityChunks.length,
      neighborhoodDensity,
      expansionScore: Number((
        passageChunks.length * 0.4 + 
        highQualityChunks.length * 0.4 + 
        neighborhoodDensity * 0.2
      ).toFixed(2)),
    };
  }

  _calculateNeighborhoodDensity(chunks) {
    const passageChunks = chunks
      .filter(c => c.type === 'passage' && c.sentenceStart !== undefined)
      .sort((a, b) => a.sentenceStart - b.sentenceStart);
    
    if (passageChunks.length < 2) return 0;
    
    let totalGaps = 0;
    for (let i = 1; i < passageChunks.length; i++) {
      const gap = passageChunks[i].sentenceStart - passageChunks[i-1].sentenceEnd;
      totalGaps += gap;
    }
    
    const avgGap = totalGaps / (passageChunks.length - 1);
    return Math.max(0, 1 - (avgGap / 3)); // Normalized density score
  }

  _calculateOverallExpansionStats(docStats) {
    const allStats = Array.from(docStats.values());
    
    return {
      totalDocuments: allStats.length,
      avgChunksPerDoc: allStats.reduce((sum, s) => sum + s.totalChunks, 0) / allStats.length,
      avgQualityPerDoc: allStats.reduce((sum, s) => sum + s.avgQuality, 0) / allStats.length,
      topExpansionDocs: allStats
        .sort((a, b) => b.expansionPotential.expansionScore - a.expansionPotential.expansionScore)
        .slice(0, 5)
        .map(s => ({ title: s.title, score: s.expansionPotential.expansionScore })),
    };
  }
}
