/* ── Farmer AI — app.js ── */

const API_BASE = "http://127.0.0.1:8000";

// ── State ──────────────────────────────────────────────────────────────────

let language = localStorage.getItem("farmerAI_lang") || "en";
let sessionId = localStorage.getItem("farmerAI_session");
let pendingDislikeChatId = null;
let isWaiting = false;

// Generate session ID once, persist it
if (!sessionId) {
  sessionId = crypto.randomUUID();
  localStorage.setItem("farmerAI_session", sessionId);
}

// ── Element refs ───────────────────────────────────────────────────────────

const navRail = document.getElementById("navRail");
const chatWindow = document.getElementById("chatWindow");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");
const typingIndicator = document.getElementById("typingIndicator");
const langToggle = document.getElementById("langToggle");
const themeToggle = document.getElementById("themeToggle");
const themeIcon = document.getElementById("themeIcon");
const modalOverlay = document.getElementById("modalOverlay");
const modalCancel = document.getElementById("modalCancel");
const modalOptions = document.getElementById("modalOptions");
const welcomeText = document.getElementById("welcomeText");
const welcomeTime = document.getElementById("welcomeTime");
const inputHint = document.getElementById("inputHint");
const weatherOverlay = document.getElementById("weatherOverlay");
const weatherCloseBtn = document.getElementById("weatherCloseBtn");
const weatherGrid = document.getElementById("weatherGrid");
const weatherModalTitle = document.getElementById("weatherModalTitle");
const weatherRefreshNote = document.getElementById("weatherRefreshNote");
const weatherSearch = document.getElementById("weatherSearch");
const weatherDistrictCount = document.getElementById("weatherDistrictCount");
const marketOverlay = document.getElementById("marketOverlay");
const marketCloseBtn = document.getElementById("marketCloseBtn");
const marketTableWrap = document.getElementById("marketTableWrap");
const marketModalTitle = document.getElementById("marketModalTitle");
const marketRefreshNote = document.getElementById("marketRefreshNote");
const marketSearch = document.getElementById("marketSearch");
const marketRecordCount = document.getElementById("marketRecordCount");
const newChatBtn = document.getElementById("newChatBtn");
const homeBackBtn = document.getElementById("homeBackBtn");
const sidebarHistoryList = document.getElementById("sidebarHistoryList");
const sidebarHistoryLabel = document.getElementById("sidebarHistoryLabel");
const diagnoseUploadBtn = document.getElementById("diagnoseUploadBtn");
const diagnoseCameraBtn = document.getElementById("diagnoseCameraBtn");
const diagnoseFileInput = document.getElementById("diagnoseFileInput");
const diagnoseCameraInput = document.getElementById("diagnoseCameraInput");
const diagnoseScanStage = document.getElementById("diagnoseScanStage");
const diagnoseScanPreview = document.getElementById("diagnoseScanPreview");
const diagnoseCardTitle = document.getElementById("diagnoseCardTitle");
const diagnoseCardDesc = document.getElementById("diagnoseCardDesc");
const diagnoseUploadLabel = document.getElementById("diagnoseUploadLabel");
const diagnoseCameraLabel = document.getElementById("diagnoseCameraLabel");

// Home / settings / navigation refs
const homeScreen = document.getElementById("homeScreen");
const chatScreen = document.getElementById("chatScreen");
const settingsBtn = document.getElementById("settingsBtn");
const settingsPanel = document.getElementById("settingsPanel");
const darkModeLabel = document.getElementById("darkModeLabel");
const languageLabel = document.getElementById("languageLabel");
const openChatBtn = document.getElementById("openChatBtn");
const openWeatherCardBtn = document.getElementById("openWeatherCardBtn");
const openMarketCardBtn = document.getElementById("openMarketCardBtn");
const chatCardLiveLabel = document.getElementById("chatCardLiveLabel");
const weatherCardLiveLabel = document.getElementById("weatherCardLiveLabel");
const marketCardLiveLabel = document.getElementById("marketCardLiveLabel");
const homeTitle = document.getElementById("homeTitle");
const chatCardTitle = document.getElementById("chatCardTitle");
const chatCardDesc = document.getElementById("chatCardDesc");
const weatherCardTitle = document.getElementById("weatherCardTitle");
const weatherCardDesc = document.getElementById("weatherCardDesc");
const weatherCardCta = document.getElementById("weatherCardCta");
const marketCardTitle = document.getElementById("marketCardTitle");
const marketCardDesc = document.getElementById("marketCardDesc");
const marketCardCta = document.getElementById("marketCardCta");


// ── i18n strings ───────────────────────────────────────────────────────────

const i18n = {
  en: {
    placeholder: "Ask a question...",
    inputHint: "Press Enter to send",
    welcome: "Hello! I'm Farmer AI, your agricultural assistant for Gujarat. Ask me anything about crops, weather, farming practices, or government schemes.",
    modalTitle: "What went wrong?",
    wrong_info: "Wrong information",
    wrong_language: "Wrong language",
    irrelevant: "Irrelevant answer",
    other: "Other",
    cancel: "Cancel",
    helpful: "👍 Helpful",
    not_helpful: "👎 Not helpful",
    kb_badge: "Knowledge Base",
    llm_badge: "AI Reasoning",
    weather_badge: "Weather API",
    leaf_badge: "Leaf Diagnosis",
    mixed_badge: "AI + Knowledge Base",
    error_chat: "Sorry, I'm temporarily unavailable. Please try again in a moment.",
    history_loading: "Loading history...",
    history_empty: "No past conversations yet",
    history_error: "Could not load history",
    history_delete: "Delete this conversation",
    history_delete_confirm: "Delete this conversation permanently? This cannot be undone.",
    history_delete_error: "Could not delete this conversation. Please try again.",
    weather_title: "Gujarat Weather",
    weather_error: "Unable to load",
    weather_humidity: "Humidity",
    weather_rain_now: "Rain now",
    weather_rain_today: "Rain today",
    weather_tomorrow_label: "Tomorrow",
    weather_tomorrow_temp_range: "{min}-{max}°C",
    weather_tomorrow_rain: "{mm}mm rain",
    weather_refresh_note: "Auto-refreshing every 10 min",
    weather_districts_all: "{count} districts",
    weather_districts_filtered: "{visible} of {total} districts",
    weather_no_districts_match: 'No districts matching "{query}"',
    weather_search_placeholder: "Search district...",
    diagnose_card_title: "Diagnose Disease",
    diagnose_card_desc: "Upload or capture a leaf photo to detect crop diseases.",
    diagnose_upload_label: "Upload",
    diagnose_camera_label: "Camera",
    home_ask_placeholder: "Type your question…",
    home_title: "Your fields. Your language. Real answers.",
    chat_card_title: "Ask Farmer AI",
    chat_card_desc: "Crops, schemes, soil, pests — ask anything in English or Gujarati.",
    chat_card_live: "English · auto",
    weather_card_title: "Weather",
    weather_card_desc: "Live temperature, humidity and rainfall for every district.",
    weather_card_cta: "View dashboard →",
    weather_card_live: "Auto-updates every 10 min",
    market_card_title: "Crop Market Prices",
    market_card_desc: "Live mandi prices for Gujarat's major crops, by category.",
    market_card_cta: "Check prices →",
    market_card_live: "Auto-updates every 30 min",
    diagnose_home_card_title: "Diagnose a Leaf",
    diagnose_home_card_desc: "Snap a photo, get the disease + remedy.",
    diagnose_home_card_cta: "Scan a leaf →",
    diagnose_home_card_live: "AI vision",
    schemes_card_title: "Scheme Finder",
    schemes_card_desc: "Subsidies & schemes you can apply to.",
    schemes_card_cta: "Browse schemes →",
    schemes_card_live: "Government",
    schemes_modal_title: "Government schemes for Gujarat farmers",
    schemes_modal_sub: "Tap any scheme to read details. Eligibility checker coming soon.",
    schemes_modal_footer: "Last reviewed: June 2026 · Links go to official portals",
    scheme_action_portal: "Open official portal",
    scheme_action_ask: "Ask AI about this",
    market_title: "Crop Market Prices",
    market_refresh_note: "Live mandi prices · data.gov.in",
    market_search_placeholder: "Search any crop or mandi...",
    market_loading: "Loading prices...",
    market_error: "Unable to load prices right now",
    market_empty: "Select a category above to see live mandi prices",
    market_no_results: "No mandi prices reported for this category today — this often happens when a crop is out of season (e.g. wheat is a winter crop, so prices thin out by summer). Try another category or check back closer to harvest time.",
    market_records_count: "{count} records",
    market_col_commodity: "Commodity",
    market_col_variety: "Variety",
    market_col_market: "Mandi",
    market_col_district: "District",
    market_col_grade: "Grade",
    market_col_date: "Date",
    market_col_min: "Min ₹/qtl",
    market_col_max: "Max ₹/qtl",
    market_col_modal: "Modal ₹/qtl",
    dark_mode_label: "Dark mode",
    language_label: "Language",
    home_back_label: "Home",
    new_chat_label: "New chat",
  },
  gu: {
    placeholder: "પ્રશ્ન પૂછો...",
    inputHint: "મોકલવા Enter દબાવો",
    welcome: "નમસ્તે! હું Farmer AI છું, ગુજરાત માટેનો તમારો કૃષિ સહાયક. પાક, હવામાન, ખેતી અથવા સરકારી યોજનાઓ વિશે પૂછો.",
    modalTitle: "શું ખોટું થયું?",
    wrong_info: "ખોટી માહિતી",
    wrong_language: "ખોટી ભાષા",
    irrelevant: "અસંગત જવાબ",
    other: "અન્ય",
    cancel: "રદ કરો",
    helpful: "👍 ઉપયોગી",
    not_helpful: "👎 ઉપયોગી નથી",
    kb_badge: "જ્ઞાન આધાર",
    llm_badge: "AI તર્ક",
    weather_badge: "હવામાન સેવા",
    leaf_badge: "પાન નિદાન",
    mixed_badge: "AI + જ્ઞાન આધાર",
    error_chat: "માફ કરશો, હું અત્યારે ઉપલબ્ધ નથી. થોડી વાર પછી ફરી પ્રયાસ કરો.",
    history_loading: "ઇતિહાસ લોડ થઈ રહ્યો છે...",
    history_empty: "હજુ સુધી કોઈ જૂની વાતચીત નથી",
    history_error: "ઇતિહાસ લોડ કરી શકાયો નહીં",
    history_delete: "આ વાતચીત કાઢી નાખો",
    history_delete_confirm: "આ વાતચીત કાયમ માટે કાઢી નાખવી છે? આ પાછું લાવી શકાશે નહીં.",
    history_delete_error: "આ વાતચીત કાઢી શકાઈ નહીં. ફરી પ્રયાસ કરો.",
    weather_title: "ગુજરાત હવામાન",
    weather_error: "લોડ કરી શકાયું નથી",
    weather_humidity: "ભેજ",
    weather_rain_now: "હાલમાં વરસાદ",
    weather_rain_today: "આજે વરસાદ",
    weather_tomorrow_label: "આવતીકાલે",
    weather_tomorrow_temp_range: "{min}-{max}°C",
    weather_tomorrow_rain: "{mm}mm વરસાદ",
    weather_refresh_note: "દર 10 મિનિટે રિફ્રેશ થાય છે",
    weather_search_placeholder: "જિલ્લો શોધો...",
    weather_districts_all: "{count} જિલ્લા",
    weather_districts_filtered: "{total} માંથી {visible} જિલ્લા",
    weather_no_districts_match: '"{query}" સાથે મેળ ખાતો કોઈ જિલ્લો નથી',
    diagnose_card_title: "રોગનું નિદાન કરો",
    diagnose_card_desc: "પાકના રોગો ઓળખવા માટે પાનનો ફોટો અપલોડ કરો અથવા લો.",
    diagnose_upload_label: "અપલોડ",
    diagnose_camera_label: "કેમેરા",
    home_ask_placeholder: "તમારો પ્રશ્ન ટાઈપ કરો…",
    home_title: "તમારા ખેતર. તમારી ભાષા. સાચા જવાબ.",
    chat_card_title: "Farmer AI ને પૂછો",
    chat_card_desc: "પાક, યોજનાઓ, માટી, જીવાતો — અંગ્રેજી અથવા ગુજરાતીમાં કંઈપણ પૂછો.",
    chat_card_live: "ગુજરાતી · ઓટો",
    weather_card_title: "હવામાન",
    weather_card_desc: "દરેક જિલ્લા માટે જીવંત તાપમાન, ભેજ અને વરસાદ.",
    weather_card_cta: "ડેશબોર્ડ જુઓ →",
    weather_card_live: "દર 10 મિનિટે ઓટો-અપડેટ",
    market_card_title: "પાક બજાર ભાવ",
    market_card_desc: "ગુજરાતના મુખ્ય પાકોના જીવંત મંડી ભાવ, શ્રેણી પ્રમાણે.",
    market_card_cta: "ભાવ જુઓ →",
    market_card_live: "દર 30 મિનિટે ઓટો-અપડેટ",
    diagnose_home_card_title: "પાનનું નિદાન કરો",
    diagnose_home_card_desc: "ફોટો લો, રોગ અને ઉપાય મેળવો.",
    diagnose_home_card_cta: "પાન સ્કેન કરો →",
    diagnose_home_card_live: "AI દ્રષ્ટિ",
    schemes_card_title: "યોજના શોધક",
    schemes_card_desc: "તમે અરજી કરી શકો તેવી સબસિડી અને યોજનાઓ.",
    schemes_card_cta: "યોજનાઓ જુઓ →",
    schemes_card_live: "સરકારી",
    schemes_modal_title: "ગુજરાતના ખેડૂતો માટે સરકારી યોજનાઓ",
    schemes_modal_sub: "વિગતો વાંચવા માટે કોઈપણ યોજના પર ટૅપ કરો. પાત્રતા તપાસનાર ટૂંક સમયમાં આવી રહ્યું છે.",
    schemes_modal_footer: "છેલ્લે સમીક્ષા: જૂન 2026 · લિંક્સ સત્તાવાર પોર્ટલ પર જાય છે",
    scheme_action_portal: "સત્તાવાર પોર્ટલ ખોલો",
    scheme_action_ask: "AI ને આ વિશે પૂછો",
    market_title: "પાક બજાર ભાવ",
    market_refresh_note: "જીવંત મંડી ભાવ · data.gov.in",
    market_search_placeholder: "કોઈપણ પાક અથવા મંડી શોધો...",
    market_loading: "ભાવ લોડ થઈ રહ્યા છે...",
    market_error: "અત્યારે ભાવ લોડ કરી શકાયા નથી",
    market_empty: "જીવંત મંડી ભાવ જોવા માટે ઉપર શ્રેણી પસંદ કરો",
    market_no_results: "આજે આ શ્રેણી માટે મંડી ભાવ ઉપલબ્ધ નથી — આ ઘણીવાર ત્યારે થાય છે જ્યારે પાક મોસમ બહાર હોય (દા.ત. ઘઉં શિયાળાનો પાક છે, તેથી ઉનાળામાં ભાવ ઓછા જોવા મળે છે). બીજી શ્રેણી પસંદ કરો અથવા લણણીના સમય નજીક ફરી તપાસો.",
    market_records_count: "{count} રેકોર્ડ",
    market_col_commodity: "પાક",
    market_col_variety: "જાત",
    market_col_market: "મંડી",
    market_col_district: "જિલ્લો",
    market_col_grade: "ગ્રેડ",
    market_col_date: "તારીખ",
    market_col_min: "ઓછો ₹/ક્વિન્ટલ",
    market_col_max: "વધુ ₹/ક્વિન્ટલ",
    market_col_modal: "સામાન્ય ₹/ક્વિન્ટલ",
    dark_mode_label: "ડાર્ક મોડ",
    language_label: "ભાષા",
    home_back_label: "હોમ",
    new_chat_label: "નવી ચેટ",
  },
};

function t(key) { return i18n[language][key] || i18n.en[key] || key; }

// District names come from the backend in English (the canonical key used
// for caching/lookups) — mirrors Backend/services.py's
// GUJARAT_DISTRICT_GUJARATI_NAMES so the weather dashboard can display the
// Gujarati script name instead of the English key when language is "gu".
const districtNamesGu = {
  "ahmedabad": "અમદાવાદ",
  "gandhinagar": "ગાંધીનગર",
  "anand": "આણંદ",
  "kheda": "ખેડા",
  "mehsana": "મહેસાણા",
  "patan": "પાટણ",
  "sabarkantha": "સાબરકાંઠા",
  "aravalli": "અરવલ્લી",
  "surat": "સુરત",
  "tapi": "તાપી",
  "navsari": "નવસારી",
  "valsad": "વલસાડ",
  "dang": "ડાંગ",
  "bharuch": "ભરૂચ",
  "vadodara": "વડોદરા",
  "chhota udaipur": "છોટા ઉદેપુર",
  "dahod": "દાહોદ",
  "panchmahals": "પંચમહાલ",
  "mahisagar": "મહીસાગર",
  "rajkot": "રાજકોટ",
  "jamnagar": "જામનગર",
  "morbi": "મોરબી",
  "surendranagar": "સુરેન્દ્રનગર",
  "botad": "બોટાદ",
  "amreli": "અમરેલી",
  "bhavnagar": "ભાવનગર",
  "junagadh": "જૂનાગઢ",
  "porbandar": "પોરબંદર",
  "gir somnath": "ગીર સોમનાથ",
  "kutch": "કચ્છ",
  "banaskantha": "બનાસકાંઠા",
  "narmada": "નર્મદા",
  "devbhoomi dwarka": "દેવભૂમિ દ્વારકા",
};

function districtDisplayName(districtKey) {
  const key = (districtKey || "").toLowerCase();
  if (language === "gu" && districtNamesGu[key]) return districtNamesGu[key];
  return districtKey;
}

// ── Theme ──────────────────────────────────────────────────────────────────

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  themeIcon.textContent = theme === "dark" ? "☀️" : "🌙";
  themeToggle.setAttribute("aria-checked", theme === "dark" ? "true" : "false");
  localStorage.setItem("farmerAI_theme", theme);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme");
  applyTheme(current === "dark" ? "light" : "dark");
}

// Restore saved theme
applyTheme(localStorage.getItem("farmerAI_theme") || "light");

themeToggle.addEventListener("click", toggleTheme);

// ── Language ──────────────────────────────────────────────────────────────

function applyLanguage(lang) {
  language = lang;
  localStorage.setItem("farmerAI_lang", lang);
  langToggle.querySelectorAll(".segmented-option").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.lang === lang);
  });
  chatInput.placeholder = t("placeholder");
  inputHint.textContent = t("inputHint");
  welcomeText.textContent = t("welcome");
  // Update modal option labels if visible
  updateModalLabels();
  weatherModalTitle.textContent = t("weather_title");
  weatherRefreshNote.textContent = t("weather_refresh_note");
  if (weatherSearch) weatherSearch.placeholder = t("weather_search_placeholder");
  if (diagnoseCardTitle) diagnoseCardTitle.textContent = t("diagnose_card_title");
  if (diagnoseCardDesc) diagnoseCardDesc.textContent = t("diagnose_card_desc");
  if (diagnoseUploadLabel) diagnoseUploadLabel.textContent = t("diagnose_upload_label");
  if (diagnoseCameraLabel) diagnoseCameraLabel.textContent = t("diagnose_camera_label");
  if (homeAskInput) homeAskInput.placeholder = t("home_ask_placeholder");
  const chatHeaderOnlineLabel = document.getElementById("chatHeaderOnlineLabel");
  if (chatHeaderOnlineLabel) chatHeaderOnlineLabel.textContent = t("chat_card_live");
  if (weatherOverlay.classList.contains("open") && lastWeatherData) {
    renderWeatherCards(lastWeatherData);
  }
  // Home screen text
  darkModeLabel.textContent = t("dark_mode_label");
  languageLabel.textContent = t("language_label");
  document.getElementById("homeBackLabel").textContent = t("home_back_label");
  document.getElementById("newChatLabel").textContent = t("new_chat_label");
  homeTitle.textContent = t("home_title");
  chatCardTitle.textContent = t("chat_card_title");
  chatCardDesc.textContent = t("chat_card_desc");
  chatCardLiveLabel.textContent = t("chat_card_live");
  weatherCardTitle.textContent = t("weather_card_title");
  weatherCardDesc.textContent = t("weather_card_desc");
  weatherCardCta.textContent = t("weather_card_cta");
  weatherCardLiveLabel.textContent = t("weather_card_live");
  marketCardTitle.textContent = t("market_card_title");
  marketCardDesc.textContent = t("market_card_desc");
  marketCardCta.textContent = t("market_card_cta");
  marketCardLiveLabel.textContent = t("market_card_live");
  // Diagnose home card (separate IDs from the chat-sidebar diagnose card,
  // which still owns `diagnoseCardTitle`/`diagnoseCardDesc`).
  const dT = document.getElementById("homeDiagnoseTitle");
  const dD = document.getElementById("homeDiagnoseDesc");
  const dC = document.getElementById("homeDiagnoseCta");
  const dL = document.getElementById("diagnoseCardLiveLabel");
  if (dT) dT.textContent = t("diagnose_home_card_title");
  if (dD) dD.textContent = t("diagnose_home_card_desc");
  if (dC) dC.textContent = t("diagnose_home_card_cta");
  if (dL) dL.textContent = t("diagnose_home_card_live");
  // Schemes home card
  const sT = document.getElementById("schemesCardTitle");
  const sD = document.getElementById("schemesCardDesc");
  const sC = document.getElementById("schemesCardCta");
  const sL = document.getElementById("schemesCardLiveLabel");
  if (sT) sT.textContent = t("schemes_card_title");
  if (sD) sD.textContent = t("schemes_card_desc");
  if (sC) sC.textContent = t("schemes_card_cta");
  if (sL) sL.textContent = t("schemes_card_live");
  // Schemes modal chrome + re-render list if open, so switching language
  // while the modal is visible updates everything in place.
  const smT = document.getElementById("schemesModalTitle");
  const smS = document.getElementById("schemesModalSub");
  const smF = document.getElementById("schemesModalFooter");
  if (smT) smT.textContent = t("schemes_modal_title");
  if (smS) smS.textContent = t("schemes_modal_sub");
  if (smF) smF.textContent = t("schemes_modal_footer");
  if (schemesOverlay && schemesOverlay.classList.contains("open")) renderSchemes();
  marketModalTitle.textContent = t("market_title");
  marketRefreshNote.textContent = t("market_refresh_note");
  marketSearch.placeholder = t("market_search_placeholder");
  if (marketOverlay.classList.contains("open")) {
    if (lastMarketRecords) renderMarketTable(lastMarketRecords);
    else renderMarketEmptyState();
  }
}

function updateModalLabels() {
  document.getElementById("modalTitle").textContent = t("modalTitle");
  document.getElementById("modalCancel").textContent = t("cancel");
  modalOptions.querySelectorAll(".modal-option").forEach(btn => {
    btn.textContent = t(btn.dataset.reason);
  });
}

langToggle.addEventListener("click", (e) => {
  const btn = e.target.closest(".segmented-option");
  if (!btn) return;
  applyLanguage(btn.dataset.lang);
});

// ── Utilities ─────────────────────────────────────────────────────────────

function formatTime(date) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function scrollToBottom() {
  chatWindow.scrollTo({ top: chatWindow.scrollHeight, behavior: "smooth" });
}

function setWaiting(val) {
  isWaiting = val;
  chatInput.disabled = val;
  sendBtn.disabled = val;
  typingIndicator.classList.toggle("visible", val);
  if (val) scrollToBottom();
}

// ── Render a user bubble ───────────────────────────────────────────────────

function appendUserMessage(text) {
  const msg = document.createElement("div");
  msg.className = "message user-message";
  msg.innerHTML = `
    <div class="avatar"><svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor" aria-hidden="true"><path d="M12 12c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm0 2c-3.33 0-10 1.67-10 5v2h20v-2c0-3.33-6.67-5-10-5z"/></svg></div>
    <div class="bubble">
      <p>${escapeHtml(text)}</p>
      <span class="timestamp">${formatTime(new Date())}</span>
    </div>
  `;
  chatWindow.appendChild(msg);
  scrollToBottom();
}

// ── Preview the uploaded leaf photo inside the diagnose card ──────────────

let diagnoseScanPreviewUrl = null;

function showDiagnoseScanPreview(file) {
  if (diagnoseScanPreviewUrl) URL.revokeObjectURL(diagnoseScanPreviewUrl);
  diagnoseScanPreviewUrl = URL.createObjectURL(file);
  diagnoseScanPreview.src = diagnoseScanPreviewUrl;
  diagnoseScanStage.classList.add("has-image");
}

// ── Render an AI bubble ───────────────────────────────────────────────────

// Map backend's source_type field to a {label, css-modifier} pair.
// Backend stores four real values: "knowledge_base" (KB grounded),
// "weather_api" (live Open-Meteo data injected), "leaf_diagnosis" (ML
// classifier + Groq remedy), and "llm_reasoning" (Groq from general
// training only). Previously the frontend treated everything except
// knowledge_base as "AI Reasoning", which mislabeled every weather and
// every leaf-diagnosis response — defeating the whole point of telling
// farmers where the answer came from.
function badgeFor(sourceType) {
  switch (sourceType) {
    case "knowledge_base":
      return { text: t("kb_badge"), cls: "source-badge--kb" };
    case "weather_api":
      return { text: t("weather_badge"), cls: "source-badge--weather" };
    case "leaf_diagnosis":
      return { text: t("leaf_badge"), cls: "source-badge--leaf" };
    case "mixed":
      return { text: t("mixed_badge"), cls: "source-badge--mixed" };
    default:
      return { text: t("llm_badge"), cls: "source-badge--llm" };
  }
}

function appendAiMessage(text, chatId, sourceType) {
  const { text: badgeText, cls: badgeCls } = badgeFor(sourceType);

  const msg = document.createElement("div");
  msg.className = "message ai-message";
  msg.dataset.chatId = chatId;
  msg.innerHTML = `
    <div class="avatar"><svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 20 C4 11 10 4 20 4 C20 14 13 20 4 20 Z" fill="currentColor" fill-opacity="0.18" stroke="currentColor"/><path d="M4 20 C9 15 14 10 20 4"/></svg></div>
    <div class="bubble">
      <span class="source-badge ${badgeCls}">${badgeText}</span>
      <div class="ai-text">${formatAiText(text)}</div>
      <span class="timestamp">${formatTime(new Date())}</span>
      <div class="feedback-row">
        <button class="feedback-btn like-btn" data-chat-id="${chatId}" title="${t("helpful")}">
          ${t("helpful")}
        </button>
        <button class="feedback-btn dislike-btn" data-chat-id="${chatId}" title="${t("not_helpful")}">
          ${t("not_helpful")}
        </button>
      </div>
    </div>
  `;
  chatWindow.appendChild(msg);

  // Attach feedback handlers
  msg.querySelector(".like-btn").addEventListener("click", () => submitFeedback(chatId, 1, null, msg));
  msg.querySelector(".dislike-btn").addEventListener("click", () => openDislikeModal(chatId));

  scrollToBottom();
}

// ── Render an error bubble ────────────────────────────────────────────────

function appendErrorMessage(text) {
  const msg = document.createElement("div");
  msg.className = "message ai-message";
  msg.innerHTML = `
    <div class="avatar"><svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 20 C4 11 10 4 20 4 C20 14 13 20 4 20 Z" fill="currentColor" fill-opacity="0.18" stroke="currentColor"/><path d="M4 20 C9 15 14 10 20 4"/></svg></div>
    <div class="bubble">
      <p style="color:#EF4444">${escapeHtml(text)}</p>
      <span class="timestamp">${formatTime(new Date())}</span>
    </div>
  `;
  chatWindow.appendChild(msg);
  scrollToBottom();
}

// ── Chat API call ─────────────────────────────────────────────────────────

async function sendMessage() {
  const query = chatInput.value.trim();
  if (!query || isWaiting) return;

  chatInput.value = "";
  appendUserMessage(query);
  setWaiting(true);

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, query, language }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    appendAiMessage(data.response, data.chat_id, data.source_type);
  } catch (err) {
    appendErrorMessage(t("error_chat"));
    console.error("Chat error:", err);
  } finally {
    setWaiting(false);
  }
}

// ── Disease diagnosis API call ────────────────────────────────────────────

async function diagnoseImage(file) {
  if (isWaiting) return;

  if (!chatScreen.classList.contains("visible")) {
    showChat();
  }

  showDiagnoseScanPreview(file);
  appendUserMessage(file.name || "Leaf photo");
  setWaiting(true);

  try {
    const formData = new FormData();
    formData.append("image", file);
    formData.append("session_id", sessionId);
    formData.append("language", language);
    formData.append("top_k", "3");

    const res = await fetch(`${API_BASE}/diagnose`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    appendAiMessage(data.response, data.chat_id, data.source_type);
  } catch (err) {
    const msg = (err.message && !err.message.startsWith("HTTP")) ? err.message : t("error_chat");
    appendErrorMessage(msg);
    console.error("Diagnose error:", err);
  } finally {
    setWaiting(false);
  }
}

// ── Feedback API call ─────────────────────────────────────────────────────

async function submitFeedback(chatId, score, reason, msgEl) {
  // Disable both buttons immediately
  if (msgEl) {
    msgEl.querySelectorAll(".feedback-btn").forEach(b => b.disabled = true);
    const likeBtn = msgEl.querySelector(".like-btn");
    const dislikeBtn = msgEl.querySelector(".dislike-btn");
    if (score === 1) likeBtn?.classList.add("active-like");
    if (score === -1) dislikeBtn?.classList.add("active-dislike");
  }

  try {
    const body = { chat_id: chatId, score };
    if (reason) body.reason = reason;

    const res = await fetch(`${API_BASE}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
  } catch (err) {
    console.error("Feedback error:", err);
    // Re-enable buttons if it failed
    if (msgEl) {
      msgEl.querySelectorAll(".feedback-btn").forEach(b => {
        b.disabled = false;
        b.classList.remove("active-like", "active-dislike");
      });
    }
  }
}

// ── Dislike modal ─────────────────────────────────────────────────────────

function openDislikeModal(chatId) {
  pendingDislikeChatId = chatId;
  updateModalLabels();
  modalOverlay.classList.add("open");
}

function closeModal() {
  modalOverlay.classList.remove("open");
  pendingDislikeChatId = null;
}

modalOptions.addEventListener("click", async (e) => {
  const btn = e.target.closest(".modal-option");
  if (!btn || !pendingDislikeChatId) return;

  const reason = btn.dataset.reason;
  const chatId = pendingDislikeChatId;
  closeModal();

  // Find the message element for this chatId
  const msgEl = chatWindow.querySelector(`[data-chat-id="${chatId}"]`);
  await submitFeedback(chatId, -1, reason, msgEl);
});

modalCancel.addEventListener("click", closeModal);
modalOverlay.addEventListener("click", (e) => {
  if (e.target === modalOverlay) closeModal();
});

// ── Load chat history on startup ──────────────────────────────────────────

async function loadHistory() {
  try {
    const res = await fetch(`${API_BASE}/chat/history?session_id=${sessionId}`);
    if (!res.ok) return;
    const rows = await res.json();
    rows.forEach(row => {
      appendUserMessage(row.query);
      appendAiMessage(row.response, row.chat_id, row.source_type);
    });
  } catch (_) {
    // Silently ignore — history is a nice-to-have, not critical
  }
}

// ── New chat ──────────────────────────────────────────────────────────────

function startNewChat() {
  // Generate a fresh session id so /chat/history on next load won't pull
  // up this conversation again. Old rows stay in the DB under the old
  // session_id — harmless, just orphaned.
  sessionId = crypto.randomUUID();
  localStorage.setItem("farmerAI_session", sessionId);

  // Remove every message except the original welcome message node, so the
  // cached welcomeText/welcomeTime element references (used elsewhere,
  // e.g. on language switch) stay valid instead of pointing to a
  // recreated, detached element.
  Array.from(chatWindow.children).forEach(child => {
    if (child.id !== "welcomeMsg") child.remove();
  });
  welcomeText.textContent = t("welcome");
  welcomeTime.textContent = formatTime(new Date());
  renderHistoryList();
}

newChatBtn.addEventListener("click", startNewChat);

// ── Input events ──────────────────────────────────────────────────────────

sendBtn.addEventListener("click", sendMessage);

chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// ── XSS protection ────────────────────────────────────────────────────────

function escapeHtml(value) {
  // Market price fields (min_price, max_price, modal_price) come back from
  // the API as raw numbers, not strings — calling .replace() directly on
  // those throws. Coerce to a string first so this works for any input.
  return String(value ?? "—")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// ── Format AI responses for display ──────────────────────────────────────
// Groq replies with a light Markdown style that varies response to
// response: **bold** section headings like "**Crop Details:**", numbered
// points like "1. **Soil**: medium-to-deep black soil.", and single-
// asterisk bullets like "* Botanical name: ... * Family: Malvaceae".
// Sections and bullets often run together in the same response with no
// separator besides the markdown itself. This converts all of that into
// safe HTML — every run on text content goes through escapeHtml() FIRST,
// so nothing Groq (or a user) sends can ever inject real HTML; only our
// own heading/bullet/line-break tags get added afterward, on top of
// already-escaped text.
function formatAiText(rawText) {
  const escaped = escapeHtml(rawText);

  const withLineBreaks = escaped
    // Break before a bolded section heading like "**Crop Details:**" —
    // this can appear anywhere in the text (often right after the
    // previous section's last bullet, with no other separator), so this
    // rule isn't anchored to start-of-text or preceded by whitespace only.
    .replace(/\s*(\*\*[^*]+:\*\*)/g, "\n$1")
    // Break "1. **Label**: detail 2. **Label**: detail" into separate
    // numbered points by inserting a line break before each "N. " that
    // starts a new point (but not numbers that are just part of a value,
    // like "75-100 mm").
    .replace(/(\s)(\d{1,2}\.\s+\*\*)/g, "$1\n$2")
    // Break "* Point one. * Point two." into separate bullet lines.
    // Requires a space after the "*" and a space/start-of-text before it,
    // so this doesn't trip on "**bold**" (no space after the first "*")
    // or on a literal multiplication/escaped asterisk mid-word.
    .replace(/(\s)\*\s+(?!\*)/g, "$1\u0001BULLET\u0001")
    // Also split off a trailing closing remark after the last point
    // (e.g. "...90x45 cm. These are just a few key points...") so it
    // doesn't get glued onto that point's line.
    .replace(/([.!])\s+(These are|Let me know|Feel free|If you have)/g, "$1\n$2");

  // Bold: **text** -> <strong>text</strong> (after heading/bullet
  // splitting above, so "**" markers are still intact for this step)
  const withBold = withLineBreaks.replace(
    /\*\*(.+?)\*\*/g,
    "<strong>$1</strong>"
  );

  // Restore the bullet marker placeholder as an actual line break now that
  // bold conversion (which also uses *) is done, so the two don't collide.
  const withBulletBreaks = withBold.split("\u0001BULLET\u0001").join("\n• ");

  // Turn each resulting line into its own line. Numbered points and
  // bullets get the .ai-point treatment (slight indent); a line that's
  // ONLY a bolded heading (nothing else on it) gets .ai-heading for
  // visual separation from the list items under it; everything else
  // stays a plain line.
  const lines = withBulletBreaks.split("\n").map((line) => line.trim()).filter(Boolean);

  return lines
    .map((line) => {
      if (/^<strong>[^<]+<\/strong>$/.test(line)) {
        return `<span class="ai-heading">${line}</span>`;
      }
      return /^(\d{1,2}\.\s|•\s)/.test(line)
        ? `<span class="ai-point">${line}</span>`
        : `<span>${line}</span>`;
    })
    .join("");
}

// ── Weather dashboard ─────────────────────────────────────────────────────

let lastWeatherData = null;
let weatherRefreshTimer = null;
const WEATHER_REFRESH_MS = 10 * 60 * 1000; // 10 minutes; aligned with backend's WEATHER_CACHE_MINUTES so each refresh actually pulls fresh data instead of returning the same cached row

// Maps Open-Meteo's WMO weather_code to a representative emoji icon.
// Codes follow the standard WMO scale (0 = clear sky ... 99 = severe storm).
function weatherIconFor(code) {
  if (code === 0) return "☀️";
  if (code >= 1 && code <= 3) return "⛅";
  if (code === 45 || code === 48) return "🌫️";
  if (code >= 51 && code <= 57) return "🌦️";
  if (code >= 61 && code <= 67) return "🌧️";
  if (code >= 71 && code <= 77) return "🌨️";
  if (code >= 80 && code <= 82) return "🌧️";
  if (code === 85 || code === 86) return "🌨️";
  if (code >= 95 && code <= 99) return "⛈️";
  return "🌡️";
}

function weatherCardHtml(data) {
  const icon = weatherIconFor(data.weather_code);
  // Tomorrow-forecast footer — only render when the backend actually
  // supplied the forecast fields (some older cached rows might not).
  // Guards on `!= null` so a legitimate 0mm or 0°C still renders.
  let tomorrowRow = "";
  if (data.temp_max_tomorrow_c != null && data.temp_min_tomorrow_c != null) {
    const tempRange = t("weather_tomorrow_temp_range")
      .replace("{min}", Math.round(data.temp_min_tomorrow_c))
      .replace("{max}", Math.round(data.temp_max_tomorrow_c));
    const rainBit = data.rainfall_tomorrow_mm != null
      ? ` · ${t("weather_tomorrow_rain").replace("{mm}", data.rainfall_tomorrow_mm)}`
      : "";
    tomorrowRow = `
      <div class="weather-card-tomorrow">
        <span class="weather-tomorrow-label">${t("weather_tomorrow_label")}</span>
        <span class="weather-tomorrow-values">${tempRange}${rainBit}</span>
      </div>
    `;
  }
  return `
    <div class="weather-card" data-district="${escapeHtml(data.district.toLowerCase())}">
      <div class="weather-card-top">
        <span class="weather-card-district">${escapeHtml(districtDisplayName(data.district))}</span>
        <span class="weather-card-icon">${icon}</span>
      </div>
      <div class="weather-card-temp">${data.temperature_c}°C</div>
      <div class="weather-card-stats">
        <div class="weather-stat-row"><span>💧 ${t("weather_humidity")}</span><span>${data.humidity_percent}%</span></div>
        <div class="weather-stat-row"><span>🌧️ ${t("weather_rain_now")}</span><span>${data.rainfall_now_mm}mm</span></div>
        <div class="weather-stat-row"><span>☔ ${t("weather_rain_today")}</span><span>${data.rainfall_today_mm}mm</span></div>
      </div>
      ${tomorrowRow}
    </div>
  `;
}

function renderWeatherSkeletons(count = 12) {
  weatherGrid.innerHTML = Array.from({ length: count }, () => `
    <div class="weather-card-skeleton">
      <div class="skeleton-line" style="height:14px;width:60%;"></div>
      <div class="skeleton-line" style="height:36px;width:50%;margin-top:12px;"></div>
      <div class="skeleton-line" style="height:11px;width:100%;margin-top:10px;"></div>
      <div class="skeleton-line" style="height:11px;width:100%;"></div>
      <div class="skeleton-line" style="height:11px;width:100%;"></div>
    </div>
  `).join("");
}

function renderWeatherCards(districts) {
  if (!districts || districts.length === 0) {
    weatherGrid.innerHTML = `<div class="weather-card weather-card-error">${t("weather_error")}</div>`;
    weatherDistrictCount.textContent = "";
    return;
  }
  weatherGrid.innerHTML = districts.map(weatherCardHtml).join("");
  updateWeatherDistrictCount(districts.length, districts.length);
  // Re-apply current search filter if any
  const q = weatherSearch ? weatherSearch.value.trim().toLowerCase() : "";
  if (q) filterWeatherCards(q);
}

function updateWeatherDistrictCount(visible, total) {
  if (visible === total) {
    weatherDistrictCount.textContent = t("weather_districts_all").replace("{count}", total);
  } else {
    weatherDistrictCount.textContent = t("weather_districts_filtered")
      .replace("{visible}", visible)
      .replace("{total}", total);
  }
}

function filterWeatherCards(query) {
  const cards = weatherGrid.querySelectorAll(".weather-card[data-district]");
  let visible = 0;
  cards.forEach(card => {
    const match = card.dataset.district.includes(query);
    card.style.display = match ? "" : "none";
    if (match) visible++;
  });
  // Remove old no-results message if present
  const prev = weatherGrid.querySelector(".weather-no-results");
  if (prev) prev.remove();
  if (visible === 0 && query) {
    const msg = document.createElement("div");
    msg.className = "weather-no-results";
    msg.textContent = t("weather_no_districts_match").replace("{query}", query);
    weatherGrid.appendChild(msg);
  }
  if (lastWeatherData) updateWeatherDistrictCount(visible, lastWeatherData.length);
}

async function fetchAllWeather() {
  try {
    const res = await fetch(`${API_BASE}/weather/all`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    lastWeatherData = data.districts || [];
    renderWeatherCards(lastWeatherData);
  } catch (err) {
    console.error("Weather dashboard error:", err);
    if (!lastWeatherData) {
      weatherGrid.innerHTML = `<div class="weather-card weather-card-error">${t("weather_error")}</div>`;
    }
  }
}

function openWeatherDashboard() {
  weatherModalTitle.textContent = t("weather_title");
  weatherRefreshNote.textContent = t("weather_refresh_note");
  weatherOverlay.classList.add("open");
  if (weatherSearch) weatherSearch.value = "";
  if (!lastWeatherData) renderWeatherSkeletons(12);
  else renderWeatherCards(lastWeatherData);
  fetchAllWeather();
  if (!weatherRefreshTimer) {
    weatherRefreshTimer = setInterval(fetchAllWeather, WEATHER_REFRESH_MS);
  }
}

function closeWeatherDashboard() {
  weatherOverlay.classList.remove("open");
  if (weatherRefreshTimer) {
    clearInterval(weatherRefreshTimer);
    weatherRefreshTimer = null;
  }
}

openWeatherCardBtn.addEventListener("click", openWeatherDashboard);
weatherCloseBtn.addEventListener("click", closeWeatherDashboard);
weatherOverlay.addEventListener("click", (e) => {
  if (e.target === weatherOverlay) closeWeatherDashboard();
});

weatherSearch.addEventListener("input", (e) => {
  filterWeatherCards(e.target.value.trim().toLowerCase());
});

// ── Market prices dashboard ──────────────────────────────────────────────
// Single flat table — every crop, every mandi. Search box filters
// `lastMarketRecords` client-side, no second fetch needed.

let lastMarketRecords = null;       // most recently rendered records array (unfiltered)

function renderMarketLoading() {
  marketTableWrap.innerHTML = `<div class="market-table-loading">${t("market_loading")}</div>`;
}

function renderMarketEmptyState() {
  marketTableWrap.innerHTML = `<div class="market-table-empty">${t("market_empty")}</div>`;
  marketRecordCount.textContent = "";
}

function renderMarketError() {
  marketTableWrap.innerHTML = `<div class="market-table-error">${t("market_error")}</div>`;
  marketRecordCount.textContent = "";
}

function marketTableRowHtml(rec) {
  const district = (rec.district || "—").toLowerCase();
  const market = (rec.market || "—").toLowerCase();
  const commodity = (rec.commodity || "—").toLowerCase();
  return `
    <tr data-district="${escapeHtml(district)}" data-market="${escapeHtml(market)}" data-commodity="${escapeHtml(commodity)}">
      <td class="market-commodity-cell">${escapeHtml(rec.commodity || "—")}</td>
      <td>${escapeHtml(rec.variety || "—")}</td>
      <td>${escapeHtml(rec.market || "—")}</td>
      <td>${escapeHtml(rec.district ? districtDisplayName(rec.district) : "—")}</td>
      <td>${escapeHtml(rec.grade || "—")}</td>
      <td>${escapeHtml(rec.arrival_date || "—")}</td>
      <td class="market-price-cell">${escapeHtml(rec.min_price ?? "—")}</td>
      <td class="market-price-cell">${escapeHtml(rec.max_price ?? "—")}</td>
      <td class="market-modal-price-cell">${escapeHtml(rec.modal_price ?? "—")}</td>
    </tr>
  `;
}

function marketTableHtml(records) {
  return `
    <table class="market-table">
      <thead>
        <tr>
          <th>${t("market_col_commodity")}</th>
          <th>${t("market_col_variety")}</th>
          <th>${t("market_col_market")}</th>
          <th>${t("market_col_district")}</th>
          <th>${t("market_col_grade")}</th>
          <th>${t("market_col_date")}</th>
          <th>${t("market_col_min")}</th>
          <th>${t("market_col_max")}</th>
          <th>${t("market_col_modal")}</th>
        </tr>
      </thead>
      <tbody>
        ${records.map(marketTableRowHtml).join("")}
      </tbody>
    </table>
  `;
}

// Stable hierarchical sort: Commodity → Variety → District. Done once
// per render (cheap for ~400 rows) using localeCompare so Gujarati script
// and English both order naturally without ad-hoc collation rules.
function sortMarketRecords(records) {
  const norm = v => (v || "").toString();
  return records.slice().sort((a, b) =>
    norm(a.commodity).localeCompare(norm(b.commodity)) ||
    norm(a.variety).localeCompare(norm(b.variety)) ||
    norm(a.district).localeCompare(norm(b.district))
  );
}

function renderMarketTable(records) {
  const all = sortMarketRecords(records || []);

  if (all.length === 0) {
    marketTableWrap.innerHTML = `<div class="market-table-empty">${t("market_no_results")}</div>`;
    marketRecordCount.textContent = "";
    return;
  }

  marketTableWrap.innerHTML = marketTableHtml(all);
  marketRecordCount.textContent = t("market_records_count").replace("{count}", all.length);
}

function matchesQuery(rec, query) {
  return (
    (rec.commodity || "").toLowerCase().includes(query) ||
    (rec.market || "").toLowerCase().includes(query) ||
    (rec.district || "").toLowerCase().includes(query) ||
    (rec.variety || "").toLowerCase().includes(query)
  );
}

function applyMarketSearch(query) {
  const all = lastMarketRecords || [];
  if (!query) {
    renderMarketTable(all);
    return;
  }
  renderMarketTable(all.filter(rec => matchesQuery(rec, query)));
}

async function fetchAllMarketPrices() {
  renderMarketLoading();
  try {
    const res = await fetch(`${API_BASE}/market-price/all`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    lastMarketRecords = data.records || [];
    const q = (marketSearch && marketSearch.value.trim().toLowerCase()) || "";
    applyMarketSearch(q);
  } catch (err) {
    console.error("Market fetch error:", err);
    renderMarketError();
  }
}

let marketSearchDebounce = null;
marketSearch.addEventListener("input", (e) => {
  const query = e.target.value.trim().toLowerCase();
  clearTimeout(marketSearchDebounce);
  marketSearchDebounce = setTimeout(() => applyMarketSearch(query), 120);
});

function openMarketDashboard() {
  marketModalTitle.textContent = t("market_title");
  marketRefreshNote.textContent = t("market_refresh_note");
  marketOverlay.classList.add("open");
  if (marketSearch) marketSearch.value = "";
  fetchAllMarketPrices();
}

function closeMarketDashboard() {
  marketOverlay.classList.remove("open");
}

openMarketCardBtn.addEventListener("click", openMarketDashboard);
marketCloseBtn.addEventListener("click", closeMarketDashboard);
marketOverlay.addEventListener("click", (e) => {
  if (e.target === marketOverlay) closeMarketDashboard();
});

// ── Home / chat navigation ───────────────────────────────────────────────

function showHome() {
  homeScreen.classList.remove("hidden");
  chatScreen.classList.remove("visible");
  navRail.classList.remove("rail-on-chat");
}

function showChat(prefillQuery) {
  homeScreen.classList.add("hidden");
  chatScreen.classList.add("visible");
  navRail.classList.add("rail-on-chat");
  scrollToBottom();
  renderHistoryList();

  if (prefillQuery) {
    chatInput.value = prefillQuery;
    sendMessage();
  }
}

openChatBtn.addEventListener("click", () => showChat());

// Diagnose home card: routes the user to the chat screen and immediately
// pops the OS file picker. Same backend flow as the chat-sidebar Upload
// button — no duplicate plumbing, the home card is just a more findable
// entry point for the feature.
const openDiagnoseCardBtn = document.getElementById("openDiagnoseCardBtn");
if (openDiagnoseCardBtn) {
  openDiagnoseCardBtn.addEventListener("click", () => {
    startNewChat();
    showChat();
    // Slight delay so the chat screen transition completes before the
    // native picker steals focus — otherwise some browsers swallow the click.
    setTimeout(() => diagnoseFileInput.click(), 180);
  });
}

// Schemes home card → open schemes modal.
const openSchemesCardBtn = document.getElementById("openSchemesCardBtn");
const schemesOverlay = document.getElementById("schemesOverlay");
const schemesCloseBtn = document.getElementById("schemesCloseBtn");
const schemesList = document.getElementById("schemesList");

// Static seed data. Swap with /schemes endpoint when backend lands; the
// modal render code reads from this array regardless of source.
// `domain` keys the left-stripe colour: income | insurance | credit |
// soil | state.
const SCHEMES = [
  {
    domain: "income",
    title: "PM-KISAN",
    title_gu: "PM-KISAN",
    tag: "Income support",
    tag_gu: "આવક સહાય",
    desc: "₹6,000/year direct cash transfer to small & marginal farmer families, in three ₹2,000 instalments.",
    desc_gu: "નાના અને સીમાંત ખેડૂત પરિવારોને દર વર્ષે ₹6,000 સીધી રોકડ સહાય, ત્રણ ₹2,000ના હપતામાં.",
    link: "https://pmkisan.gov.in",
    // Pre-filled question covers the six things every farmer actually asks
    // about a scheme: am I eligible, how much, when paid, how to apply,
    // documents, and why applications get rejected.
    question: "Explain PM-KISAN for a Gujarat farmer: who is eligible and who is excluded, exactly how much money I get and when each instalment is paid, the step-by-step process to register (both online and at the CSC), the documents I need, how to check my payment status, and the most common reasons applications get rejected or instalments are stopped.",
    question_gu: "ગુજરાતના ખેડૂત માટે PM-KISAN સમજાવો: કોણ પાત્ર છે અને કોણ બાકાત છે, મને કેટલા પૈસા મળે છે અને દરેક હપતો ક્યારે ચૂકવાય છે, ઑનલાઇન અને CSC બંને રીતે રજિસ્ટ્રેશનની પ્રક્રિયા, જરૂરી દસ્તાવેજો, મારી ચુકવણીની સ્થિતિ કેવી રીતે તપાસવી, અને અરજી નકારવા અથવા હપતા બંધ થવાના સામાન્ય કારણો.",
  },
  {
    domain: "insurance",
    title: "Pradhan Mantri Fasal Bima Yojana",
    title_gu: "પ્રધાનમંત્રી ફસલ બીમા યોજના",
    tag: "Crop insurance",
    tag_gu: "પાક વીમો",
    desc: "Premium-subsidised cover against drought, flood, pest, hail and post-harvest losses for notified crops.",
    desc_gu: "દુષ્કાળ, પૂર, જીવાત, કરા અને કાપણી પછીના નુકસાન સામે પ્રીમિયમ-સબસિડીવાળું વીમા કવર.",
    link: "https://pmfby.gov.in",
    question: "Explain PMFBY for a Gujarat farmer: which crops and risks are covered, what premium I pay as a percentage of sum insured for kharif and rabi crops, the cut-off dates for enrolment, how claims are calculated and paid, the documents and process to file a claim after a crop loss, and the main reasons claims get rejected or delayed.",
    question_gu: "ગુજરાતના ખેડૂત માટે PMFBY સમજાવો: કયા પાક અને જોખમો આવરી લેવાય છે, ખરીફ અને રવિ પાક માટે વીમા રકમના ટકા તરીકે મારે કેટલું પ્રીમિયમ ભરવું પડે, નોંધણીની છેલ્લી તારીખો, દાવાની ગણતરી અને ચુકવણી કેવી રીતે થાય છે, પાક નુકસાન પછી દાવો કરવાની પ્રક્રિયા અને દસ્તાવેજો, અને દાવા નકારવા કે વિલંબના મુખ્ય કારણો.",
  },
  {
    domain: "credit",
    title: "Kisan Credit Card (KCC)",
    title_gu: "કિસાન ક્રેડિટ કાર્ડ (KCC)",
    tag: "Credit",
    tag_gu: "ધિરાણ",
    desc: "Short-term credit up to ₹3 lakh for crop inputs, at 7% interest with prompt-repayment reduction to 4%.",
    desc_gu: "પાક ઇનપુટ્સ માટે ₹3 લાખ સુધીની ટૂંકા-ગાળાની લોન, 7% વ્યાજ પર — સમયસર ચુકવણીમાં ઘટાડીને 4%.",
    link: "https://www.myscheme.gov.in/schemes/kcc",
    question: "Explain Kisan Credit Card (KCC) for a Gujarat farmer: who qualifies, the maximum credit I can get based on my land size and crops, the effective interest rate after the prompt-repayment subvention, what I can use the credit for (inputs, livestock, allied activities), the documents and process to apply through my bank, how renewal works, and what happens if I miss a repayment.",
    question_gu: "ગુજરાતના ખેડૂત માટે કિસાન ક્રેડિટ કાર્ડ (KCC) સમજાવો: કોણ પાત્ર છે, મારી જમીન અને પાક પ્રમાણે મહત્તમ કેટલી લોન મળે, સમયસર-ચુકવણી સબવેન્શન પછી અસરકારક વ્યાજ દર, હું તેનો ઉપયોગ શેના માટે કરી શકું (ઇનપુટ્સ, પશુપાલન, સંબંધિત પ્રવૃત્તિ), મારી બેન્ક દ્વારા અરજીની પ્રક્રિયા અને દસ્તાવેજો, રિન્યુઅલ કેવી રીતે થાય, અને હપતો ચૂકી જાઉં તો શું થાય.",
  },
  {
    domain: "soil",
    title: "Soil Health Card",
    title_gu: "સોઇલ હેલ્થ કાર્ડ",
    tag: "Soil testing",
    tag_gu: "માટી પરીક્ષણ",
    desc: "Free lab analysis of your field's soil with nutrient-specific fertilizer recommendations, valid for 3 years.",
    desc_gu: "તમારા ખેતરની માટીનું મફત લેબ વિશ્લેષણ — પોષક-તત્ત્વ આધારિત ખાતર સૂચનો સાથે, 3 વર્ષ માટે માન્ય.",
    link: "https://soilhealth.dac.gov.in",
    question: "Explain the Soil Health Card scheme for a Gujarat farmer: how to get my soil tested (the sampling procedure, where to submit samples, cost if any), what the card actually tells me about my soil's nutrients and pH, how to read the fertilizer recommendations, how often I should retest, and how I can use the report to lower my fertilizer cost without hurting yield.",
    question_gu: "ગુજરાતના ખેડૂત માટે સોઇલ હેલ્થ કાર્ડ યોજના સમજાવો: માટીનું પરીક્ષણ કેવી રીતે કરાવવું (નમૂના લેવાની પદ્ધતિ, ક્યાં સબમિટ કરવા, કિંમત), કાર્ડ ખરેખર મને માટીના પોષક તત્ત્વો અને pH વિશે શું જણાવે છે, ખાતરની ભલામણો કેવી રીતે વાંચવી, ફરી ક્યારે ટેસ્ટ કરાવવો, અને ઉત્પાદન ઘટાડ્યા વગર ખાતર ખર્ચ ઘટાડવા માટે રિપોર્ટનો ઉપયોગ કેવી રીતે કરવો.",
  },
  {
    domain: "state",
    title: "iKhedut Portal (Gujarat)",
    title_gu: "આઇ-ખેડૂત પોર્ટલ (ગુજરાત)",
    tag: "State subsidies",
    tag_gu: "રાજ્ય સબસિડી",
    desc: "Single-window application for Gujarat-specific subsidies on seeds, equipment, irrigation and horticulture.",
    desc_gu: "બીજ, સાધનો, સિંચાઈ અને બાગાયત માટેની ગુજરાત-વિશિષ્ટ સબસિડી માટેનું એક-બારી અરજી પ્લેટફોર્મ.",
    link: "https://ikhedut.gujarat.gov.in",
    question: "Explain the iKhedut portal for a Gujarat farmer: which categories of subsidies are available (seeds, drip/sprinkler irrigation, farm equipment, horticulture, livestock), how the application window and selection process works, the documents I need to upload, how subsidies are disbursed, how to track my application status, and tips to improve my chances of approval given limited annual quotas.",
    question_gu: "ગુજરાતના ખેડૂત માટે આઇ-ખેડૂત પોર્ટલ સમજાવો: કઈ સબસિડી શ્રેણીઓ ઉપલબ્ધ છે (બીજ, ડ્રિપ/સ્પ્રિંકલર સિંચાઈ, ખેત-સાધનો, બાગાયત, પશુપાલન), અરજી વિન્ડો અને પસંદગી પ્રક્રિયા કેવી રીતે કામ કરે છે, મારે કયા દસ્તાવેજો અપલોડ કરવા, સબસિડી કેવી રીતે વિતરિત થાય, મારી અરજીની સ્થિતિ કેવી રીતે ટ્રૅક કરવી, અને મર્યાદિત વાર્ષિક ક્વોટા જોતાં મંજૂરીની તકો વધારવાની ટિપ્સ.",
  },
  {
    domain: "income",
    title: "PM-KMY (Pension)",
    title_gu: "PM-KMY (પેન્શન)",
    tag: "Pension",
    tag_gu: "પેન્શન",
    desc: "₹3,000/month pension after age 60 for small & marginal farmers. ₹55–200/month contribution depending on entry age.",
    desc_gu: "નાના અને સીમાંત ખેડૂતો માટે 60 વર્ષ પછી દર મહિને ₹3,000 પેન્શન. પ્રવેશ ઉંમર પ્રમાણે ₹55–200/મહિને યોગદાન.",
    link: "https://maandhan.in/pmkmy",
    question: "Explain PM-KMY (Kisan Maandhan Yojana) for a Gujarat farmer: who is eligible, the exact monthly contribution required at different entry ages between 18 and 40, how the government's matching contribution works, when and how the ₹3,000/month pension is paid after age 60, what happens to the corpus if I die or want to exit early, and how to enrol at my nearest CSC.",
    question_gu: "ગુજરાતના ખેડૂત માટે PM-KMY (કિસાન માનધન યોજના) સમજાવો: કોણ પાત્ર છે, 18 થી 40 વચ્ચે અલગ-અલગ પ્રવેશ ઉંમરે જરૂરી ચોક્કસ માસિક યોગદાન, સરકારનું મેચિંગ યોગદાન કેવી રીતે કામ કરે છે, 60 વર્ષ પછી ₹3,000/મહિને પેન્શન ક્યારે અને કેવી રીતે ચૂકવાય છે, મારા મૃત્યુ કે વહેલા બહાર નીકળવા પર કોર્પસનું શું થાય છે, અને નજીકના CSC પર નોંધણી કેવી રીતે કરવી.",
  },
];

function renderSchemes() {
  if (!schemesList) return;
  const gu = language === "gu";
  const portalLabel = t("scheme_action_portal");
  const askLabel = t("scheme_action_ask");
  schemesList.innerHTML = SCHEMES.map((s, i) => {
    const title = gu ? s.title_gu : s.title;
    const tag = gu ? s.tag_gu : s.tag;
    const desc = gu ? s.desc_gu : s.desc;
    return `
    <div class="scheme-card" data-domain="${escapeHtml(s.domain)}" data-scheme-i="${i}">
      <span class="scheme-stripe" aria-hidden="true"></span>
      <div class="scheme-body">
        <div class="scheme-title">
          ${escapeHtml(title)}
          <span class="scheme-tag">${escapeHtml(tag)}</span>
        </div>
        <div class="scheme-desc">${escapeHtml(desc)}</div>
        <div class="scheme-actions">
          <a class="scheme-action scheme-action-portal"
             href="${escapeHtml(s.link)}" target="_blank" rel="noopener noreferrer">
            ${escapeHtml(portalLabel)}
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
              stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M7 17 L17 7"/><path d="M8 7 H17 V16"/>
            </svg>
          </a>
          <button class="scheme-action scheme-action-ask" type="button" data-ask-i="${i}">
            ${escapeHtml(askLabel)}
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
              stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M5 12 H19"/><path d="M13 6 L19 12 L13 18"/>
            </svg>
          </button>
        </div>
      </div>
    </div>
  `;
  }).join("");

  // Single delegated handler for all "Ask AI" buttons — closes the modal,
  // navigates to chat, pre-fills the input with the comprehensive question,
  // focuses so the user can edit (add their district, plot size, etc.)
  // before sending. We deliberately don't auto-send: the question is a
  // strong default, but every farmer has slightly different context worth
  // adding.
  schemesList.querySelectorAll(".scheme-action-ask").forEach(btn => {
    btn.addEventListener("click", (e) => {
      const idx = Number(btn.dataset.askI);
      const scheme = SCHEMES[idx];
      if (!scheme) return;
      closeSchemesModal();
      askAboutScheme(scheme);
    });
  });
}

// Starts a fresh chat session and auto-sends the scheme's pre-filled
// question. The user lands in the chat screen with their question
// already posted and the AI reply streaming in — zero extra clicks
// after pressing "Ask AI about this".
function askAboutScheme(scheme) {
  startNewChat();           // fresh session_id, wiped welcome view
  const q = language === "gu" ? scheme.question_gu : scheme.question;
  showChat(q);              // navigates to chat AND auto-sends the prefill
}

function openSchemesModal() {
  if (!schemesOverlay) return;
  renderSchemes();
  schemesOverlay.classList.add("open");
}
function closeSchemesModal() {
  if (!schemesOverlay) return;
  schemesOverlay.classList.remove("open");
}

if (openSchemesCardBtn) openSchemesCardBtn.addEventListener("click", openSchemesModal);
if (schemesCloseBtn) schemesCloseBtn.addEventListener("click", closeSchemesModal);
if (schemesOverlay) {
  schemesOverlay.addEventListener("click", (e) => {
    if (e.target === schemesOverlay) closeSchemesModal();
  });
}
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && schemesOverlay && schemesOverlay.classList.contains("open")) {
    closeSchemesModal();
  }
});
homeBackBtn.addEventListener("click", showHome);

// ── Home "ask a question" box — types straight into the card and the
// first question rides along into the chat screen ──────────────────────
const homeAskInput = document.getElementById("homeAskInput");
const homeAskSend = document.getElementById("homeAskSend");

function submitHomeAsk() {
  const query = homeAskInput.value.trim();
  if (!query) return;
  homeAskInput.value = "";
  showChat(query);
}

homeAskInput.addEventListener("click", (e) => e.stopPropagation());
homeAskInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    submitHomeAsk();
  }
});
homeAskSend.addEventListener("click", (e) => {
  e.stopPropagation();
  submitHomeAsk();
});

// ── Sidebar (New Chat + History) ────────────────────────────────────────

// Renders the History list: fetches every past conversation (newest
// first, per /chat/sessions), shows a short preview of each, and marks
// whichever one matches the CURRENT sessionId so the user can see which
// conversation they're in.
async function renderHistoryList() {
  sidebarHistoryList.innerHTML = `<div class="sidebar-history-empty">${t("history_loading")}</div>`;

  try {
    const res = await fetch(`${API_BASE}/chat/sessions`);
    if (!res.ok) throw new Error("Failed to load sessions");
    const sessions = await res.json();

    if (!sessions.length) {
      sidebarHistoryList.innerHTML = `<div class="sidebar-history-empty">${t("history_empty")}</div>`;
      return;
    }

    sidebarHistoryList.innerHTML = sessions.map(s => `
      <div class="sidebar-history-item ${s.session_id === sessionId ? "active-session" : ""}" data-session-id="${escapeHtml(s.session_id)}">
        <span class="sidebar-history-item-text">${escapeHtml(s.preview)}</span>
        <button class="sidebar-history-item-delete" data-session-id="${escapeHtml(s.session_id)}" title="${t("history_delete")}" aria-label="${t("history_delete")}">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="3 6 5 6 21 6"></polyline>
            <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"></path>
            <path d="M10 11v6"></path>
            <path d="M14 11v6"></path>
            <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"></path>
          </svg>
        </button>
      </div>
    `).join("");
  } catch (_) {
    sidebarHistoryList.innerHTML = `<div class="sidebar-history-empty">${t("history_error")}</div>`;
  }
}

// Switches the active conversation to a past session: swaps sessionId,
// clears the chat window, and replays that session's messages — reuses
// loadHistory(), the same function that restores history on page load.
async function switchToSession(targetSessionId) {
  sessionId = targetSessionId;
  localStorage.setItem("farmerAI_session", sessionId);

  Array.from(chatWindow.children).forEach(child => {
    if (child.id !== "welcomeMsg") child.remove();
  });

  await loadHistory();
  scrollToBottom();
  renderHistoryList();
}

// Deletes a session permanently (hard delete — rows are removed from the
// database, this cannot be undone). Confirms first since it's irreversible.
async function deleteSession(targetSessionId) {
  if (!window.confirm(t("history_delete_confirm"))) return;

  try {
    const res = await fetch(`${API_BASE}/chat/session/${targetSessionId}`, { method: "DELETE" });
    if (!res.ok) throw new Error("Delete failed");

    // If the deleted session was the active one, start a fresh session
    // so the user isn't left pointing at chat history that no longer exists.
    if (targetSessionId === sessionId) {
      startNewChat();
    }

    renderHistoryList();
  } catch (_) {
    window.alert(t("history_delete_error"));
  }
}

sidebarHistoryList.addEventListener("click", (e) => {
  const deleteBtn = e.target.closest(".sidebar-history-item-delete");
  if (deleteBtn) {
    e.stopPropagation();
    deleteSession(deleteBtn.dataset.sessionId);
    return;
  }

  const item = e.target.closest(".sidebar-history-item");
  if (item) {
    switchToSession(item.dataset.sessionId);
  }
});

// ── Settings dropdown ────────────────────────────────────────────────────

function openSettings() {
  settingsPanel.classList.add("open");
  settingsBtn.classList.add("active");
  settingsBtn.setAttribute("aria-expanded", "true");
}

function closeSettings() {
  settingsPanel.classList.remove("open");
  settingsBtn.classList.remove("active");
  settingsBtn.setAttribute("aria-expanded", "false");
}

settingsBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  if (settingsPanel.classList.contains("open")) closeSettings();
  else openSettings();
});

document.addEventListener("click", (e) => {
  if (!settingsPanel.contains(e.target) && e.target !== settingsBtn) {
    closeSettings();
  }
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeSettings();
});

// ── Diagnose Disease card (upload / camera) ──────────────────────────────

diagnoseUploadBtn.addEventListener("click", () => diagnoseFileInput.click());
diagnoseCameraBtn.addEventListener("click", () => diagnoseCameraInput.click());

diagnoseFileInput.addEventListener("change", () => {
  const file = diagnoseFileInput.files[0];
  if (file) diagnoseImage(file);
  diagnoseFileInput.value = "";
});

diagnoseCameraInput.addEventListener("change", () => {
  const file = diagnoseCameraInput.files[0];
  if (file) diagnoseImage(file);
  diagnoseCameraInput.value = "";
});

// ── Init ──────────────────────────────────────────────────────────────────

welcomeTime.textContent = formatTime(new Date());
applyLanguage(language);
loadHistory();
renderHistoryList();