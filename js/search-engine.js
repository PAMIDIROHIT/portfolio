/**
 * Search Engine for Portfolio
 * Handles search results generation, URL routing, and result formatting
 */

const SEARCH_ENGINE = {

  // Search categories for tab filtering
  CATEGORIES: ['all', 'projects', 'skills', 'education', 'contact', 'achievements'],

  // Main search function
  search(query, category = 'all') {
    if (!query || query.trim().length === 0) return [];
    const q = query.toLowerCase().trim();

    return KNOWLEDGE_BASE.filter(doc => {
      const matchesCategory = category === 'all' || doc.category === category || 
        (category === 'projects' && doc.category === 'project') ||
        (category === 'achievements' && doc.category === 'achievement');
      
      if (!matchesCategory) return false;

      const searchText = `${doc.title} ${doc.content} ${doc.keywords.join(' ')}`.toLowerCase();
      // Check if any word in query matches
      const words = q.split(/\s+/);
      return words.some(w => searchText.includes(w)) || 
             doc.keywords.some(k => k.includes(q) || q.includes(k));
    }).map(doc => ({
      ...doc,
      relevance: this._computeRelevance(doc, q)
    })).sort((a, b) => b.relevance - a.relevance);
  },

  _computeRelevance(doc, query) {
    let score = 0;
    const searchText = `${doc.title} ${doc.content} ${doc.keywords.join(' ')}`.toLowerCase();
    
    // Title match (highest priority)
    if (doc.title.toLowerCase().includes(query)) score += 10;
    
    // Keyword exact match
    if (doc.keywords.some(k => k.includes(query) || query.includes(k))) score += 7;
    
    // Content match
    const words = query.split(/\s+/);
    words.forEach(w => {
      const count = (searchText.match(new RegExp(w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g')) || []).length;
      score += count;
    });
    
    // Category boost for exact matches
    if (doc.category === 'project' && /project|app|build/.test(query)) score += 3;
    if (doc.category === 'skills' && /skill|tech|know/.test(query)) score += 3;

    return score;
  },

  // Format doc URL for display
  getDisplayURL(doc) {
    const urls = {
      'project-tomato-food':           'github.com/PAMIDIROHIT/Food-Delivery-MAIN1',
      'project-heart-disease':         'github.com/PAMIDIROHIT/HeartDisease-ML-FullStack',
      'project-cinematch':             'github.com/PAMIDIROHIT/cinematch-ai',
      'project-wanderlust':            'github.com/PAMIDIROHIT/Wanderlust',
      'project-nestjs-api':            'github.com/PAMIDIROHIT/nestjs-api-project',
      'project-ai-chat-pdf':           'github.com/PAMIDIROHIT/AI-search-chat-pdf-viewer',
      'project-rl-traffic':            'github.com/PAMIDIROHIT/RL',
      'project-quickcart':             'github.com/PAMIDIROHIT/QuickCart-main',
      'project-maternal-health':       'github.com/PAMIDIROHIT/Maternal_Health_Risk',
      'project-knowledge-graph':       'github.com/PAMIDIROHIT/-Knowledge-Graph-Augmented-Retrieval-System',
      'project-beyondchats':           'github.com/PAMIDIROHIT/beyondchats-assignment',
      'project-brinavv':               'github.com/PAMIDIROHIT/Brinavv_Assignment',
      'project-taskflow':              'github.com/PAMIDIROHIT/taskflow-backend-assignment',
      'project-telugu-vae':            'github.com/PAMIDIROHIT/telugu-vae-btp',
      'project-human-aligned-ai':      'github.com/PAMIDIROHIT/Human-Aligned-AI-',
      'project-tailortalk':            'github.com/PAMIDIROHIT/Tailortalk',
      'project-text-summarization':    'github.com/PAMIDIROHIT/Text_Summarization',
      'project-healthcare-prediction': 'github.com/PAMIDIROHIT/Healthcare-Disease-Prediction-Risk-Assessment-System',
      'project-codsoft':               'github.com/PAMIDIROHIT/CODSOFT',
      'contact-info':                  'linkedin.com/in/rohit-pamidi-4147771ba',
      'achievement-leetcode':          'leetcode.com/u/rohithtnsp',
      'achievement-hackathon':         'github.com/PAMIDIROHIT',
    };
    return urls[doc.id] || 'github.com/PAMIDIROHIT';
  },

  getResultURL(doc) {
    const urls = {
      'project-tomato-food':           'https://github.com/PAMIDIROHIT/Food-Delivery-MAIN1',
      'project-heart-disease':         'https://github.com/PAMIDIROHIT/HeartDisease-ML-FullStack',
      'project-cinematch':             'https://github.com/PAMIDIROHIT/cinematch-ai',
      'project-wanderlust':            'https://github.com/PAMIDIROHIT/Wanderlust',
      'project-nestjs-api':            'https://github.com/PAMIDIROHIT/nestjs-api-project',
      'project-ai-chat-pdf':           'https://github.com/PAMIDIROHIT/AI-search-chat-pdf-viewer',
      'project-rl-traffic':            'https://github.com/PAMIDIROHIT/RL',
      'project-quickcart':             'https://github.com/PAMIDIROHIT/QuickCart-main',
      'project-maternal-health':       'https://github.com/PAMIDIROHIT/Maternal_Health_Risk',
      'project-knowledge-graph':       'https://github.com/PAMIDIROHIT/-Knowledge-Graph-Augmented-Retrieval-System',
      'project-beyondchats':           'https://github.com/PAMIDIROHIT/beyondchats-assignment',
      'project-brinavv':               'https://github.com/PAMIDIROHIT/Brinavv_Assignment',
      'project-taskflow':              'https://github.com/PAMIDIROHIT/taskflow-backend-assignment',
      'project-telugu-vae':            'https://github.com/PAMIDIROHIT/telugu-vae-btp',
      'project-human-aligned-ai':      'https://github.com/PAMIDIROHIT/Human-Aligned-AI-',
      'project-tailortalk':            'https://github.com/PAMIDIROHIT/Tailortalk',
      'project-text-summarization':    'https://github.com/PAMIDIROHIT/Text_Summarization',
      'project-healthcare-prediction': 'https://github.com/PAMIDIROHIT/Healthcare-Disease-Prediction-Risk-Assessment-System',
      'project-codsoft':               'https://github.com/PAMIDIROHIT/CODSOFT',
      'contact-info':                  'https://www.linkedin.com/in/rohit-pamidi-4147771ba/',
      'achievement-leetcode':          'https://leetcode.com/u/rohithtnsp/',
      'achievement-hackathon':         'https://github.com/PAMIDIROHIT',
    };
    return urls[doc.id] || 'https://github.com/PAMIDIROHIT';
  },

  // Category icons
  getCategoryIcon(category) {
    const icons = {
      project: '🚀', skills: '💡', education: '🎓', 
      personal: '👤', contact: '📬', achievement: '🏆', research: '🔬'
    };
    return icons[category] || '📄';
  },

  // Snippet extraction (best matching excerpt)
  getSnippet(doc, query) {
    const content = doc.content;
    const q = query.toLowerCase();
    const words = q.split(/\s+/).filter(w => w.length > 2);
    
    // Find best position in content
    let bestPos = 0;
    let bestScore = -1;
    words.forEach(w => {
      const pos = content.toLowerCase().indexOf(w);
      if (pos !== -1) {
        const score = words.filter(ow => {
          const nearPos = content.toLowerCase().indexOf(ow);
          return nearPos !== -1 && Math.abs(nearPos - pos) < 200;
        }).length;
        if (score > bestScore) { bestScore = score; bestPos = pos; }
      }
    });

    // Extract snippet around best position
    const start = Math.max(0, bestPos - 40);
    const end = Math.min(content.length, bestPos + 200);
    let snippet = content.slice(start, end);
    if (start > 0) snippet = '...' + snippet;
    if (end < content.length) snippet += '...';

    // Highlight query words
    words.forEach(w => {
      const re = new RegExp(`(${w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
      snippet = snippet.replace(re, '<strong>$1</strong>');
    });

    return snippet;
  }
};

// URL query parameter utilities
function getSearchQuery() {
  const params = new URLSearchParams(window.location.search);
  return params.get('q') || '';
}

function goToSearch(query) {
  if (query.trim()) {
    window.location.href = `search.html?q=${encodeURIComponent(query.trim())}`;
  }
}
