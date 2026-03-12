/**
 * Translation Engine for Pamidi Rohit Portfolio
 * Provides comprehensive translation functionality across all pages
 */

class TranslationEngine {
  constructor() {
    this.currentLang = 'en';
    this.supportedLanguages = {
      'en': { name: 'English', flag: '🇺🇸', code: 'en' },
      'es': { name: 'Español', flag: '🇪🇸', code: 'es' },
      'fr': { name: 'Français', flag: '🇫🇷', code: 'fr' },
      'de': { name: 'Deutsch', flag: '🇩🇪', code: 'de' },
      'hi': { name: 'हिंदी', flag: '🇮🇳', code: 'hi' },
      'te': { name: 'తెలుగు', flag: '🇮🇳', code: 'te' }
    };
    this.translations = this.loadTranslations();
    this.init();
  }

  init() {
    // Load saved language preference
    const savedLang = localStorage.getItem('portfolioLanguage') || 'en';
    this.currentLang = savedLang;
    
    // Set page language
    document.documentElement.lang = this.currentLang;
    
    // Initialize Google Translate if not already loaded
    this.initializeGoogleTranslate();
    
    // Create language switcher UI
    this.createLanguageSwitcher();
    
    // Apply saved language
    this.applyTranslation(this.currentLang);
  }

  loadTranslations() {
    return {
      // Navigation & Header
      'nav.about': {
        'en': 'About',
        'es': 'Acerca de',
        'fr': 'À propos',
        'de': 'Über',
        'hi': 'परिचय',
        'te': 'గురించి'
      },
      'nav.contact': {
        'en': 'Contact',
        'es': 'Contacto',
        'fr': 'Contact',
        'de': 'Kontakt',
        'hi': 'संपर्क',
        'te': 'సంప్రదింపు'
      },
      'nav.projects': {
        'en': 'Projects',
        'es': 'Proyectos',
        'fr': 'Projets',
        'de': 'Projekte',
        'hi': 'परियोजनाएं',
        'te': 'ప్రాజెక్టులు'
      },
      'nav.home': {
        'en': 'Home',
        'es': 'Inicio',
        'fr': 'Accueil',
        'de': 'Startseite',
        'hi': 'होम',
        'te': 'హోమ్'
      },
      'nav.images': {
        'en': 'Images',
        'es': 'Imágenes',
        'fr': 'Images',
        'de': 'Bilder',
        'hi': 'चित्र',
        'te': 'చిత్రాలు'
      },
      'nav.gmail': {
        'en': 'Gmail',
        'es': 'Gmail',
        'fr': 'Gmail',
        'de': 'Gmail',
        'hi': 'Gmail',
        'te': 'Gmail'
      },

      // Search interface
      'search.placeholder': {
        'en': 'Search anything...',
        'es': 'Buscar cualquier cosa...',
        'fr': 'Rechercher quoi que ce soit...',
        'de': 'Alles durchsuchen...',
        'hi': 'कुछ भी खोजें...',
        'te': 'ఏదైనా వెతకండి...'
      },
      'search.google': {
        'en': 'Google Search',
        'es': 'Búsqueda de Google',
        'fr': 'Recherche Google',
        'de': 'Google-Suche',
        'hi': 'Google खोज',
        'te': 'Google వెతుకుట'
      },
      'search.lucky': {
        'en': 'I\'m Feeling Lucky',
        'es': 'Voy a tener suerte',
        'fr': 'J\'ai de la chance',
        'de': 'Auf gut Glück',
        'hi': 'मुझे भाग्यशाली लग रहा है',
        'te': 'నేను అదృష్టవంతుడిని అనిపిస్తోంది'
      },
      'search.results.meta': {
        'en': 'About {count} results for \'{query}\'',
        'es': 'Aproximadamente {count} resultados para \'{query}\'',
        'fr': 'Environ {count} résultats pour \'{query}\'',
        'de': 'Ungefähr {count} Ergebnisse für \'{query}\'',
        'hi': '\'{query}\' के लिए लगभग {count} परिणाम',
        'te': '\'{query}\' కోసం దాదాపు {count} ఫలితాలు'
      },
      'search.no.results': {
        'en': 'No results found',
        'es': 'No se encontraron resultados',
        'fr': 'Aucun résultat trouvé',
        'de': 'Keine Ergebnisse gefunden',
        'hi': 'कोई परिणाम नहीं मिला',
        'te': 'ఏ ఫలితాలు కనుగొనబడలేదు'
      },

      // Categories
      'category.all': {
        'en': 'All',
        'es': 'Todos',
        'fr': 'Tous',
        'de': 'Alle',
        'hi': 'सभी',
        'te': 'అన్ని'
      },
      'category.projects': {
        'en': 'Projects',
        'es': 'Proyectos',
        'fr': 'Projets',
        'de': 'Projekte',
        'hi': 'परियोजनाएं',
        'te': 'ప్రాజెక్టులు'
      },
      'category.skills': {
        'en': 'Skills',
        'es': 'Habilidades',
        'fr': 'Compétences',
        'de': 'Fähigkeiten',
        'hi': 'कौशल',
        'te': 'నైపుణ్యాలు'
      },
      'category.education': {
        'en': 'Education',
        'es': 'Educación',
        'fr': 'Éducation',
        'de': 'Bildung',
        'hi': 'शिक्षा',
        'te': 'విద్య'
      },
      'category.achievements': {
        'en': 'Achievements',
        'es': 'Logros',
        'fr': 'Réalisations',
        'de': 'Erfolge',
        'hi': 'उपलब्धियां',
        'te': 'విజయాలు'
      },
      'category.contact': {
        'en': 'Contact',
        'es': 'Contacto',
        'fr': 'Contact',
        'de': 'Kontakt',
        'hi': 'संपर्क',
        'te': 'సంప్రదింపు'
      },

      // Chat interface
      'chat.welcome.greeting': {
        'en': 'Hey there! 👋',
        'es': '¡Hola! 👋',
        'fr': 'Salut! 👋',
        'de': 'Hallo! 👋',
        'hi': 'नमस्ते! 👋',
        'te': 'హలో! 👋'
      },
      'chat.welcome.subtitle': {
        'en': 'Ask me anything about Pamidi Rohit',
        'es': 'Pregúntame cualquier cosa sobre Pamidi Rohit',
        'fr': 'Demandez-moi tout sur Pamidi Rohit',
        'de': 'Frag mich alles über Pamidi Rohit',
        'hi': 'पामिडी रोहित के बारे में मुझसे कुछ भी पूछें',
        'te': 'పామిడి రోహిత్ గురించి నన్ను ఏదైనా అడుగండి'
      },
      'chat.typing': {
        'en': 'Typing...',
        'es': 'Escribiendo...',
        'fr': 'En train de taper...',
        'de': 'Tippt...',
        'hi': 'टाइप कर रहे हैं...',
        'te': 'టైప్ చేస్తున్నాను...'
      },
      'chat.thinking': {
        'en': 'Analyzing query...',
        'es': 'Analizando consulta...',
        'fr': 'Analyse de la requête...',
        'de': 'Anfrage analysieren...',
        'hi': 'प्रश्न का विश्लेषण कर रहे हैं...',
        'te': 'ప్రశ్నను విశ్లేషిస్తున్నాను...'
      },
      'chat.new.chat': {
        'en': 'New Chat',
        'es': 'Nuevo Chat',
        'fr': 'Nouveau Chat',
        'de': 'Neuer Chat',
        'hi': 'नई चैट',
        'te': 'కొత్త చాట్'
      },

      // Knowledge panel
      'panel.education': {
        'en': 'Education',
        'es': 'Educación',
        'fr': 'Éducation',
        'de': 'Bildung',
        'hi': 'शिक्षा',
        'te': 'విద్య'
      },
      'panel.specialization': {
        'en': 'Specialization',
        'es': 'Especialización',
        'fr': 'Spécialisation',
        'de': 'Spezialisierung',
        'hi': 'विशेषज्ञता',
        'te': 'స్పెషలైజేషన్'
      },
      'panel.top.skills': {
        'en': 'Top Skills',
        'es': 'Principales Habilidades',
        'fr': 'Compétences Principales',
        'de': 'Top-Fähigkeiten',
        'hi': 'मुख्य कौशल',
        'te': 'ప్రధాన నైపుణ్యాలు'
      },

      // Language switcher
      'lang.switch': {
        'en': 'Language',
        'es': 'Idioma',
        'fr': 'Langue',
        'de': 'Sprache',
        'hi': 'भाषा',
        'te': 'భాష'
      }
    };
  }

  createLanguageSwitcher() {
    // Remove existing switcher
    const existing = document.querySelector('.lang-switcher');
    if (existing) existing.remove();

    // Create language switcher
    const switcher = document.createElement('div');
    switcher.className = 'lang-switcher';
    switcher.innerHTML = `
      <button class="lang-btn" title="${this.translate('lang.switch')}">
        <span class="lang-flag">${this.supportedLanguages[this.currentLang].flag}</span>
        <span class="lang-code">${this.currentLang.toUpperCase()}</span>
        <svg class="lang-arrow" viewBox="0 0 24 24" width="16" height="16">
          <path d="M7 10l5 5 5-5z" fill="currentColor"/>
        </svg>
      </button>
      <div class="lang-dropdown">
        ${Object.entries(this.supportedLanguages).map(([code, lang]) => `
          <button class="lang-option ${code === this.currentLang ? 'active' : ''}" 
                  data-lang="${code}">
            <span class="lang-flag">${lang.flag}</span>
            <span class="lang-name">${lang.name}</span>
            ${code === this.currentLang ? '<svg class="lang-check" viewBox="0 0 24 24" width="16"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" fill="currentColor"/></svg>' : ''}
          </button>
        `).join('')}
      </div>
    `;

    // Add styles
    this.addTranslationStyles();

    // Insert into header-right (before apps menu)
    const headerRight = document.querySelector('#header-right');
    if (headerRight) {
      headerRight.insertBefore(switcher, headerRight.children[headerRight.children.length - 2]);
    }

    // Add event listeners
    this.attachSwitcherEvents(switcher);
  }

  addTranslationStyles() {
    if (document.querySelector('#translation-styles')) return;

    const style = document.createElement('style');
    style.id = 'translation-styles';
    style.textContent = `
      /* Language Switcher */
      .lang-switcher {
        position: relative;
        margin-right: 8px;
      }

      .lang-btn {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 6px 10px;
        background: none;
        border: 1px solid #dadce0;
        border-radius: 20px;
        cursor: pointer;
        font-size: 13px;
        color: #3c4043;
        transition: all 0.15s ease;
        min-width: 70px;
        justify-content: center;
      }

      .lang-btn:hover {
        background: #f8f9fa;
        border-color: #c6c8ca;
      }

      .lang-flag {
        font-size: 14px;
        line-height: 1;
      }

      .lang-code {
        font-weight: 500;
        font-size: 12px;
      }

      .lang-arrow {
        fill: #5f6368;
        transition: transform 0.15s ease;
      }

      .lang-switcher.open .lang-arrow {
        transform: rotate(180deg);
      }

      .lang-dropdown {
        position: absolute;
        top: 100%;
        right: 0;
        width: 200px;
        background: white;
        border-radius: 8px;
        box-shadow: 0 4px 16px rgba(60,64,67,0.15);
        border: 1px solid #dadce0;
        padding: 8px 0;
        z-index: 1000;
        display: none;
        margin-top: 4px;
      }

      .lang-switcher.open .lang-dropdown {
        display: block;
      }

      .lang-option {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 16px;
        border: none;
        background: none;
        width: 100%;
        cursor: pointer;
        font-size: 14px;
        color: #3c4043;
        transition: background 0.15s ease;
        justify-content: space-between;
      }

      .lang-option:hover {
        background: #f8f9fa;
      }

      .lang-option.active {
        background: #e8f0fe;
        color: #1a73e8;
      }

      .lang-option .lang-name {
        flex: 1;
        text-align: left;
      }

      .lang-check {
        fill: #1a73e8;
        width: 16px;
        height: 16px;
      }

      /* Mobile responsive */
      @media (max-width: 768px) {
        .lang-btn {
          min-width: 60px;
          padding: 6px 8px;
        }
        
        .lang-code {
          font-size: 11px;
        }
        
        .lang-dropdown {
          width: 180px;
        }
      }

      @media (max-width: 400px) {
        .lang-btn {
          min-width: 50px;
          padding: 4px 6px;
        }
        
        .lang-flag {
          font-size: 12px;
        }
        
        .lang-code {
          display: none;
        }
      }

      /* Google Translate Override */
      .goog-te-banner-frame { display: none !important; }
      .goog-te-menu-value { color: transparent !important; }
      .goog-te-gadget { font-size: 0 !important; }
      body { top: 0px !important; }
    `;
    document.head.appendChild(style);
  }

  attachSwitcherEvents(switcher) {
    const btn = switcher.querySelector('.lang-btn');
    const dropdown = switcher.querySelector('.lang-dropdown');

    // Toggle dropdown
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      switcher.classList.toggle('open');
    });

    // Close on outside click
    document.addEventListener('click', () => {
      switcher.classList.remove('open');
    });

    // Handle language selection
    dropdown.addEventListener('click', (e) => {
      const option = e.target.closest('.lang-option');
      if (option) {
        const lang = option.dataset.lang;
        this.switchLanguage(lang);
        switcher.classList.remove('open');
      }
    });
  }

  switchLanguage(lang) {
    if (lang === this.currentLang) return;

    this.currentLang = lang;
    localStorage.setItem('portfolioLanguage', lang);
    document.documentElement.lang = lang;

    // Update UI
    this.applyTranslation(lang);
    this.createLanguageSwitcher(); // Recreate to update active state

    // Trigger Google Translate
    this.triggerGoogleTranslate(lang);
  }

  applyTranslation(lang) {
    // Translate data-translate elements
    document.querySelectorAll('[data-translate]').forEach(el => {
      const key = el.dataset.translate;
      const translation = this.translate(key, lang);
      
      if (el.tagName === 'INPUT' && el.type === 'text') {
        el.placeholder = translation;
      } else {
        el.textContent = translation;
      }
    });

    // Update page title
    const titles = {
      'en': 'PAMIDI ROHIT — Full-Stack Developer & AI Engineer',
      'es': 'PAMIDI ROHIT — Desarrollador Full-Stack e Ingeniero de IA',
      'fr': 'PAMIDI ROHIT — Développeur Full-Stack et Ingénieur IA',
      'de': 'PAMIDI ROHIT — Full-Stack-Entwickler und KI-Ingenieur',
      'hi': 'पामिडी रोहित — फुल-स्टैक डेवलपर और AI इंजीनियर',
      'te': 'పామిడి రోహిత్ — ఫుల్-స్టాక్ డెవలపర్ మరియు AI ఇంజనీర్'
    };
    document.title = titles[lang] || titles['en'];
  }

  translate(key, lang = this.currentLang) {
    const translation = this.translations[key];
    if (!translation) return key;
    return translation[lang] || translation['en'] || key;
  }

  initializeGoogleTranslate() {
    if (window.google && window.google.translate) return;

    // Add Google Translate script
    const script = document.createElement('script');
    script.src = 'https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit';
    document.head.appendChild(script);

    // Initialize function
    window.googleTranslateElementInit = () => {
      new google.translate.TranslateElement({
        pageLanguage: 'en',
        includedLanguages: 'en,es,fr,de,hi,te,zh,ja,ko,ar,pt,ru,it',
        autoDisplay: false,
        layout: google.translate.TranslateElement.InlineLayout.SIMPLE
      }, 'google_translate_element');
    };

    // Create hidden element for Google Translate
    const translateDiv = document.createElement('div');
    translateDiv.id = 'google_translate_element';
    translateDiv.style.display = 'none';
    document.body.appendChild(translateDiv);
  }

  triggerGoogleTranslate(lang) {
    setTimeout(() => {
      const googleTranslate = document.querySelector('.goog-te-combo');
      if (googleTranslate) {
        googleTranslate.value = lang;
        googleTranslate.dispatchEvent(new Event('change'));
      }
    }, 500);
  }

  // Utility method to format text with variables
  formatText(template, variables) {
    return template.replace(/\{(\w+)\}/g, (match, key) => variables[key] || match);
  }
}

// Initialize translation engine when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  window.translationEngine = new TranslationEngine();
});

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
  module.exports = TranslationEngine;
}