/**
 * RAG Engine - Retrieval-Augmented Generation from Scratch
 * Implements TF-IDF vectorization + Cosine Similarity for semantic retrieval
 * Pure JavaScript, no external dependencies
 */

class RAGEngine {
  constructor(documents) {
    this.documents = documents;
    this.vocabMap = new Map();    // term -> index
    this.idf = [];                // IDF scores per term
    this.docTFIDF = [];           // Sparse TF-IDF vectors per doc
    this._lastTopic = null;       // Tracks last response for follow-ups
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

  // ─── Index Building ────────────────────────────────────────────────────────

  _buildIndex() {
    const N = this.documents.length;
    const allTokensPerDoc = this.documents.map(doc =>
      this._tokenize(this._getDocText(doc))
    );

    // Build vocabulary
    const vocab = new Set();
    allTokensPerDoc.forEach(tokens => tokens.forEach(t => vocab.add(t)));
    let idx = 0;
    vocab.forEach(t => { this.vocabMap.set(t, idx++); });

    const V = this.vocabMap.size;

    // Compute document frequencies
    const df = new Float32Array(V);
    allTokensPerDoc.forEach(tokens => {
      const seen = new Set(tokens);
      seen.forEach(t => {
        const i = this.vocabMap.get(t);
        if (i !== undefined) df[i]++;
      });
    });

    // Compute IDF: log((N+1)/(df+1)) + 1  (smooth IDF)
    this.idf = new Float32Array(V);
    for (let i = 0; i < V; i++) {
      this.idf[i] = Math.log((N + 1) / (df[i] + 1)) + 1;
    }

    // Compute TF-IDF sparse vectors per document
    this.docTFIDF = allTokensPerDoc.map(tokens => {
      const tf = new Map();
      tokens.forEach(t => tf.set(t, (tf.get(t) || 0) + 1));
      const L = tokens.length || 1;
      const vec = new Map();
      tf.forEach((count, term) => {
        const i = this.vocabMap.get(term);
        if (i !== undefined) {
          const tfidf = (count / L) * this.idf[i];
          if (tfidf > 0) vec.set(i, tfidf);
        }
      });
      // L2 normalize
      let norm = 0;
      vec.forEach(v => norm += v * v);
      norm = Math.sqrt(norm) || 1;
      vec.forEach((v, k) => vec.set(k, v / norm));
      return vec;
    });
  }

  // ─── Query Vectorization ──────────────────────────────────────────────────

  _vectorizeQuery(query) {
    const tokens = this._tokenize(query);
    const tf = new Map();
    tokens.forEach(t => tf.set(t, (tf.get(t) || 0) + 1));
    const L = tokens.length || 1;
    const vec = new Map();

    tf.forEach((count, term) => {
      const i = this.vocabMap.get(term);
      if (i !== undefined) {
        const tfidf = (count / L) * this.idf[i];
        if (tfidf > 0) vec.set(i, tfidf);
      }
    });

    // L2 normalize
    let norm = 0;
    vec.forEach(v => norm += v * v);
    norm = Math.sqrt(norm) || 1;
    vec.forEach((v, k) => vec.set(k, v / norm));
    return vec;
  }

  // ─── Cosine Similarity ────────────────────────────────────────────────────

  _cosineSim(qVec, docVec) {
    let dot = 0;
    // Iterate over smaller vector for efficiency
    const [small, large] = qVec.size <= docVec.size ? [qVec, docVec] : [docVec, qVec];
    small.forEach((v, k) => {
      const dv = large.get(k);
      if (dv !== undefined) dot += v * dv;
    });
    return dot; // vectors are already normalized
  }

  // ─── Retrieval ────────────────────────────────────────────────────────────

  retrieve(query, topK = 4) {
    const qVec = this._vectorizeQuery(query);
    const scores = this.documents.map((doc, i) => ({
      doc,
      score: this._cosineSim(qVec, this.docTFIDF[i])
    }));

    // Also boost exact keyword matches
    const queryLower = query.toLowerCase();
    scores.forEach(s => {
      const kw = s.doc.keywords.join(' ').toLowerCase();
      if (kw.includes(queryLower) || queryLower.includes(s.doc.category)) {
        s.score += 0.15;
      }
      // Boost title matches
      if (s.doc.title.toLowerCase().includes(queryLower)) {
        s.score += 0.25;
      }
    });

    return scores
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .filter(s => s.score > 0.01);
  }

  // ─── Response Generation ──────────────────────────────────────────────────

  generateResponse(query, profile) {
    profile = profile || (typeof PROFILE !== 'undefined' ? PROFILE : {});
    const queryLower = query.toLowerCase().trim();

    // ─── Conversational Handler (catches ALL casual/social queries) ───
    const conversational = this._handleConversational(queryLower, profile);
    if (conversational) return conversational;

    // ─── Detail follow-up detection ───────────────────────────────────
    const wantsDetail = /more|detail|elaborate|explain more|tell me more|full|in.?depth|expand|go deeper|everything/.test(queryLower);
    if (wantsDetail && this._lastTopic) {
      return this._detailedResponse(this._lastTopic, profile);
    }

    // ─── Single-word / short-form query handler ────────────────────
    const shortAnswer = this._handleShortQuery(queryLower, profile);
    if (shortAnswer) return shortAnswer;

    const results = this.retrieve(query, 5);

    if (results.length === 0) {
      return this._smartFallback(query, profile);
    }

    // Intent detection
    const isAbout     = /who|about|yourself|introduce|tell me|overview|summary|rohit|pamidi/.test(queryLower);
    const isProject   = /project|build|develop|creat|work|app|application|github|repo/.test(queryLower);
    const isSkill     = /skill|know|language|technolog|expertise|proficien|stack|tool|framework/.test(queryLower);
    const isEducation = /educat|college|universit|degree|study|cgpa|gpa|iiit|btech|school|percent/.test(queryLower);
    const isContact   = /contact|email|phone|linkedin|whatsapp|reach|connect|hire|message/.test(queryLower);
    const isAchieve   = /achieve|award|leetcode|hackathon|competi|solve|problem|bah|isro|lunar/.test(queryLower);
    const isHackathon = /hackathon|bah25|bah|lunar|moon|chandrayaan|isro|space|pioneer|neurocore|agentic|smart.?city|kernelcrew|kernel.?crew/.test(queryLower);
    const isHire      = /hire|recruit|should i|good fit|team|worth|capable|can he|can rohit|is he|is rohit|recommend/.test(queryLower);
    const isFunFact   = /fun|interesting|cool|amazing|wow|surprising|random|trivia|did you know|unique|fact/.test(queryLower);

    // Route to specific formatters
    const topDocs = results.map(r => r.doc);

    if (isContact) {
      this._lastTopic = { type: 'contact', docs: topDocs };
      return this._formatContact(profile);
    }
    if (isHire) {
      this._lastTopic = { type: 'hire', docs: topDocs };
      return this._formatHire(profile);
    }
    if (isHackathon) {
      this._lastTopic = { type: 'hackathon', docs: topDocs };
      return this._formatHackathon(topDocs, queryLower);
    }
    if (isAbout && !isProject && !isSkill) {
      this._lastTopic = { type: 'about', docs: topDocs };
      return this._formatAbout(profile, topDocs);
    }
    if (isEducation) {
      this._lastTopic = { type: 'education', docs: topDocs };
      return this._formatEducation(profile, topDocs);
    }
    if (isAchieve) {
      this._lastTopic = { type: 'achievements', docs: topDocs };
      return this._formatAchievements(topDocs);
    }
    if (isFunFact) {
      this._lastTopic = { type: 'funfact', docs: topDocs };
      return this._formatFunFact(profile);
    }
    if (isProject) {
      this._lastTopic = { type: 'projects', docs: topDocs };
      return this._formatProjects(topDocs, queryLower);
    }
    if (isSkill) {
      this._lastTopic = { type: 'skills', docs: topDocs };
      return this._formatSkills(profile, topDocs);
    }

    // Default: use top retrieved context
    this._lastTopic = { type: 'general', docs: topDocs };
    return this._formatGeneral(topDocs, query);
  }

  // ─── Fuzzy Match Helper ─────────────────────────────────────────
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
    if (stripped.length <= 2 || /^[^a-z]*$/i.test(stripped)) {
      return `I didn't quite catch that! 😅 Try asking me something like:\n- "Who is Rohit?"\n- "What projects has he built?"\n- "What are his skills?"\n- "How to contact Rohit?"`;
    }

    // ── 1. GOOD MORNING / EVENING / NIGHT (specific, check before fuzzy greetings) ──
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
      return `I'm **Rohit AI** — a custom RAG (Retrieval-Augmented Generation) chatbot built entirely from scratch by **Pamidi Rohit**.\n\n**How I work:**\n- I use **TF-IDF vectorization** to index Rohit's entire portfolio\n- When you ask a question, I compute **cosine similarity** to find the most relevant information\n- I then generate a structured, readable response\n\nI'm not powered by ChatGPT or Gemini — I'm a **100% custom-built AI** that demonstrates Rohit's NLP and information retrieval skills.\n\nAsk me anything about Rohit!`;
    }

    // ── 3.5. HELP check (must be before fuzzy greetings since "help" fuzzy-matches "helo") ──
    if (/^(help|what can you (do|tell)|how (to |do i )?use|commands|options|features|what do you know|what questions|capabilities|what should i ask|examples|sample questions|guide me)/i.test(q)) {
      return `## How to Use Rohit AI\n\nYou can ask me **anything** about Pamidi Rohit. Here are some examples:\n\n**🧑 About Rohit**\n- "Who is Pamidi Rohit?"\n- "Tell me about his background"\n- "What are his strengths?"\n\n**💻 Projects (22+)**\n- "What are all his projects?"\n- "Tell me about the Tomato food delivery app"\n- "What ML projects has he built?"\n- "Show me his hackathon projects"\n\n**🛠 Skills**\n- "What programming languages does he know?"\n- "What is his tech stack?"\n- "Does he know Docker/AWS/MLOps?"\n\n**🎓 Education & Achievements**\n- "Where does Rohit study?"\n- "How many LeetCode problems solved?"\n- "Tell me about his ISRO hackathon"\n\n**📬 Contact & Hiring**\n- "How can I reach Rohit?"\n- "Should I hire Rohit?"\n- "Is he available for internships?"\n\n**🎲 Fun**\n- "Tell me something interesting about Rohit"\n- "Fun facts"\n\nJust type naturally — I understand casual questions, typos, and follow-ups!`;
    }

    // ── 4. GREETINGS (with typo tolerance — now after specific patterns) ──
    const greetWords = ['hello','hellow','helo','hi','hey','hii','hiii','howdy','sup','yo','heya','hola','namaste','namaskar','greetings','ola','bonjour','salut','salam','ahoy','aloha','wassup','whatsup'];
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
    if (/^(bye|goodbye|good ?bye|see you|see ya|later|take care|cya|gtg|gotta go|peace|adios|sayonara|cheers|catch you later|talk later|signing off|good night|gn)[!.\s]*$/i.test(q)) {
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
        return this._detailedResponse(this._lastTopic, profile);
      }
      return `Sure! What topic would you like me to explore? I can tell you about:\n- **Projects** — 22+ applications\n- **Skills** — Full-stack, ML/AI, DevOps\n- **Achievements** — LeetCode, hackathons\n- **Education** — IIIT Sri City\n- **Contact** — Email, LinkedIn, GitHub`;
    }

    // ── 10. COMPLIMENTS ──
    if (/^(you('re| are) (smart|good|great|amazing|awesome|helpful|impressive|clever|brilliant)|good (bot|ai|job)|nice (bot|ai|work)|impressive|well done|bravo|wow|amazing|smart bot)[!.\s]*$/i.test(q)) {
      return `Thank you! 😊 That means a lot. I was built from scratch by Rohit using **TF-IDF vectorization** and **cosine similarity** — no external AI APIs needed.\n\nThis chatbot itself is a demonstration of Rohit's **NLP** and **information retrieval** engineering skills!`;
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
    if (/what (model|algorithm|technology|method)|how do you (work|function|think|operate)|powered by|built with|what('s| is) behind|architecture|how were you (made|built|trained|created)|your (technology|stack|source)|open.?source/i.test(q)) {
      return `**How Rohit AI Works:**\n\n1. **Knowledge Base** — Rohit's entire portfolio (22+ projects, skills, education, achievements) is structured as searchable documents\n2. **TF-IDF Vectorization** — Each document is converted into a mathematical vector using Term Frequency–Inverse Document Frequency\n3. **Cosine Similarity** — When you ask a question, I find the most relevant documents by computing cosine similarity between your query and all documents\n4. **Intent Detection** — I classify your query (project question? skill question? greeting?) and format the response appropriately\n5. **Conversational Layer** — I handle greetings, follow-ups, typos, and casual conversation naturally\n\nNo external AI APIs — this is a **custom RAG engine built from scratch** in JavaScript. The code is part of Rohit's portfolio itself!`;
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

  // ─── Formatters ───────────────────────────────────────────────────────────

  _formatAbout(profile, docs) {
    return `**${profile.name}** is a ${profile.degree} student at **${profile.college}** (CGPA: ${profile.cgpa}). He's a full-stack developer and ML engineer with 22+ projects, 600+ LeetCode problems solved, and experience building production-grade apps with React, Node.js, Python, and PyTorch. He also built a Lunar Hazard Detection System for ISRO's BAH25 hackathon and architected NeuroCore, a multi-agent AI smart city platform.\n\nAsk me to *tell you more* for a detailed breakdown.`;
  }

  _formatContact(profile) {
    return `You can reach Rohit at:\n\n- **Email**: [${profile.email}](mailto:${profile.email})\n- **WhatsApp**: [+91 ${profile.whatsapp}](https://wa.me/91${profile.whatsapp})\n- **LinkedIn**: [rohit-pamidi](${profile.linkedin})\n- **GitHub**: [PAMIDIROHIT](${profile.github})\n\nHe's open to internships, full-time roles, freelance projects, and open-source collaboration.`;
  }

  _formatEducation(profile, docs) {
    return `Rohit is pursuing **B.Tech in AI & Data Science** at **IIIT Sri City** (Aug 2023 -- May 2027) with a **CGPA of ${profile.cgpa}**. He scored **93.7%** in Intermediate at Sri Chaitanya Junior College (MPC stream, 2021--2023).\n\nAsk me to *elaborate* for more details about his academic journey.`;
  }

  _formatAchievements(docs) {
    return `Key achievements:\n\n- **600+ LeetCode problems** solved (DSA, DP, Graphs, Trees, Binary Search)\n- **Bharatiya Antariksh Hackathon 2025** -- Team Lunar Pioneers built an AI-powered Lunar Hazard Detection System for ISRO PS-11 using physics-informed neural networks and Chandrayaan satellite imagery\n- **NeuroCore Agentic AI** -- Team KernelCrew built a multi-agent smart city platform with 6 specialized AI agents\n- **CGPA 8.23** at IIIT Sri City, 93.7% in Intermediate\n\nSay *tell me more* about any specific achievement.`;
  }

  _formatHackathon(docs, queryLower) {
    const isLunar = /lunar|bah|moon|chandrayaan|isro|space|pioneer|ps.?11/.test(queryLower);
    const isNeuro = /neurocore|agentic|smart.?city|kernelcrew|kernel.?crew|agent/.test(queryLower);

    if (isNeuro) {
      return `**NeuroCore** is a multi-agent AI smart city platform by Team KernelCrew. It deploys **6 specialized AI agents** (Traffic, Safety, Health, Environment, Emergency, Social) that collaborate through a **Consensus Engine** for unified city-wide decisions. Built with Gemini 1.5 Pro, Vertex AI, Google Maps, React+TypeScript, Node.js, MongoDB, and Firebase. Key innovations include Neural Event Mesh, City Oracle for natural-language what-if simulations, and Privacy-by-Design with Edge AI.\n\nAsk me to *go deeper* for the full architecture.`;
    }
    if (isLunar) {
      return `**Lunar Hazard Detection System** was built for ISRO's Bharatiya Antariksh Hackathon 2025 (PS-11) by Team **Lunar Pioneers**. Rohit (Team Member-3, IIIT Sri City) helped build an AI system that detects landslides and boulders on the Moon using Chandrayaan satellite imagery (TMC, DTM, OHRC). The system uses a **Physics-Informed Neural Network** with attention-based sensor fusion, achieving pixel-wise hazard classification. Tech: PyTorch, OpenCV, Rasterio, React.js, MongoDB.\n\nSay *tell me more* for full technical details.`;
    }
    // general hackathon
    return `Rohit has participated in major hackathons:\n\n1. **BAH25 (ISRO)** -- Team Lunar Pioneers built a Lunar Hazard Detection System using physics-informed neural networks on Chandrayaan data (PS-11)\n2. **Agentic AI Competition** -- Team KernelCrew built NeuroCore, a multi-agent smart city platform with 6 AI agents\n\nAsk about either one specifically for more details.`;
  }

  _formatProjects(docs, queryLower) {
    const projectDocs = docs.filter(d => d.category === 'project');
    if (projectDocs.length === 0) return this._formatGeneral(docs, queryLower);

    const isOverview = /all|list|overview|projects/.test(queryLower);
    if (isOverview) {
      return `Rohit has built **22+ projects** across full-stack, ML/AI, and cloud:\n\n- **Tomato** -- Full-stack food delivery with Gemini AI chatbot, Redis, Socket.io\n- **CineMatch AI** -- Movie recommendations (10K+ movies, 95% accuracy)\n- **HeartDisease ML** -- 6 ML models, XGBoost, AWS deployed\n- **WanderLust** -- Airbnb-like platform with Stripe, Cloudinary\n- **NestJS API** -- Enterprise REST API with 51 unit tests\n- **AI Search Chat PDF** -- Perplexity-style SSE streaming\n- **Lunar Hazard Detection** -- BAH25 ISRO hackathon, PyTorch\n- **NeuroCore** -- Multi-agent smart city (6 AI agents)\n- **Knowledge Graph RAG** -- Graph-augmented retrieval system\n\n...and more. Ask about any specific project for details, or say *tell me more* for the full list.`;
    }

    // Format top matching project concisely
    const top = projectDocs[0];
    const snippet = top.content.slice(0, 300);
    let response = `**${top.title}**\n\n${snippet}...`;
    if (projectDocs.length > 1) {
      response += `\n\n**Related**: ${projectDocs.slice(1, 3).map(d => d.title).join(' | ')}`;
    }
    response += `\n\nSay *tell me more* for the full breakdown.`;
    return response;
  }

  _formatSkills(profile, docs) {
    const s = profile.skills;
    return `Rohit's core tech stack:\n\n- **Languages**: ${s.languages.join(', ')}\n- **Frontend**: ${s.frontend.slice(0, 4).join(', ')}\n- **Backend**: ${s.backend.join(', ')}\n- **ML/AI**: ${s.mlai.slice(0, 6).join(', ')}\n- **Databases**: ${s.databases.join(', ')}\n- **DevOps**: ${s.tools.slice(0, 4).join(', ')}\n\nHe's a full-stack developer capable of owning the entire pipeline. Say *tell me more* for detailed skill breakdowns.`;
  }

  _formatGeneral(docs, query) {
    const best = docs[0];
    if (!best) return this._unknownResponse(query);

    const snippet = best.content.slice(0, 350);
    let response = `${snippet}...`;
    const related = docs.slice(1, 3);
    if (related.length > 0) {
      response += `\n\n**Related:** ${related.map(d => d.title).join(' | ')}`;
    }
    return response;
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
}
