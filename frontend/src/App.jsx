import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { 
  FileText, Briefcase, Sparkles, AlertTriangle, CheckCircle2, 
  ChevronDown, ChevronRight, UploadCloud, RefreshCw, Send, 
  HelpCircle, Trash2, ShieldCheck, Zap, Layers, BarChart3, Clock,
  Search, Globe, Settings, LogOut, Share2, MoreVertical, Plus,
  Mic, ArrowUp, PanelLeft, Laptop, BookOpen, User, X, Check,
  FileCheck, Edit3, ClipboardList, Info, Trophy, Users, Award, Star,
  Download, Copy
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE || (typeof window !== 'undefined' && window.location.port === '5173' ? 'http://localhost:8000/api' : '/api');

export default function App() {
  // Mode Switcher: 'candidate' | 'recruiter' | 'eval'
  const [appMode, setAppMode] = useState(() => {
    return localStorage.getItem('jd_fit_app_mode') || 'candidate';
  });

  // UI & Layout State
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [showModelDropdown, setShowModelDropdown] = useState(false);
  const [showPasteDialog, setShowPasteDialog] = useState(false);
  const [jdTab, setJdTab] = useState('upload'); // 'upload' | 'paste'

  // Candidate Mode State
  const [resumeFile, setResumeFile] = useState(null);
  const [jdFiles, setJdFiles] = useState([]);
  const [pastedJds, setPastedJds] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('jd_fit_pasted_jds') || '[]');
    } catch {
      return [];
    }
  });
  const [sessionId, setSessionId] = useState(() => {
    return localStorage.getItem('jd_fit_session_id') || null;
  });
  const [selectedModel, setSelectedModel] = useState(() => {
    return localStorage.getItem('jd_fit_selected_model') || 'qwen/qwen3.8-27b';
  });
  const [availableModels, setAvailableModels] = useState([
    { id: 'qwen/qwen3.8-27b', label: 'Qwen 3.8 27B', tag: 'Reasoning' },
    { id: 'openai/gpt-oss-20b', label: 'GPT-OSS 20B', tag: 'Ultra-Fast' },
    { id: 'openai/gpt-oss-120b', label: 'GPT-OSS 120B', tag: 'Flagship 120B' }
  ]);
  const [groqConnected, setGroqConnected] = useState(true);
  const [newJdName, setNewJdName] = useState('');
  const [newJdText, setNewJdText] = useState('');
  const [isIndexing, setIsIndexing] = useState(false);
  const [indexStatus, setIndexStatus] = useState(null);
  const [loadedScope, setLoadedScope] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('jd_fit_loaded_scope') || '{"resume": null, "jds": []}');
    } catch {
      return { resume: null, jds: [] };
    }
  });

  // Recruiter Mode State
  const [recruiterJdFile, setRecruiterJdFile] = useState(null);
  const [recruiterJdText, setRecruiterJdText] = useState('');
  const [recruiterResumes, setRecruiterResumes] = useState([]);
  const [leaderboard, setLeaderboard] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('jd_fit_leaderboard') || '[]');
    } catch {
      return [];
    }
  });
  const [recruiterAnalysis, setRecruiterAnalysis] = useState(() => {
    return localStorage.getItem('jd_fit_recruiter_analysis') || '';
  });
  const [recruiterSessionId, setRecruiterSessionId] = useState(() => {
    return localStorage.getItem('jd_fit_recruiter_session_id') || null;
  });
  const [recruiterMessages, setRecruiterMessages] = useState([]);
  const [isRecruiterAnswering, setIsRecruiterAnswering] = useState(false);
  const [isRanking, setIsRanking] = useState(false);
  const [rankingStatus, setRankingStatus] = useState(null);

  // Eval Dashboard & Reliability State
  const [evalStats, setEvalStats] = useState(null);
  const [isLoadingEval, setIsLoadingEval] = useState(false);
  const [isExportingPdf, setIsExportingPdf] = useState(false);
  const [fairnessAudit, setFairnessAudit] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('jd_fit_fairness_audit') || 'null');
    } catch {
      return null;
    }
  });

  // Chat & Query State (Starts fresh on reload while keeping documents persistent)
  const [query, setQuery] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [messages, setMessages] = useState([]);
  const [expandedSources, setExpandedSources] = useState({});
  const [shareToast, setShareToast] = useState(null);
  
  // User Authentication State
  const [currentUser, setCurrentUser] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('jd_fit_current_user') || 'null');
    } catch {
      return null;
    }
  });
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [showUserDropdown, setShowUserDropdown] = useState(false);
  const [authTab, setAuthTab] = useState('login'); // 'login' | 'register'
  const [authName, setAuthName] = useState('');
  const [authEmail, setAuthEmail] = useState('');
  const [authPassword, setAuthPassword] = useState('');
  const [authRole, setAuthRole] = useState('candidate');
  const [authError, setAuthError] = useState('');

  const [history, setHistory] = useState(() => {
    try {
      const savedUser = JSON.parse(localStorage.getItem('jd_fit_current_user') || 'null');
      if (savedUser && savedUser.email) {
        return JSON.parse(localStorage.getItem(`jd_fit_user_${savedUser.email}_history`) || '[]');
      }
      return JSON.parse(localStorage.getItem('jd_fit_history_list') || '[]');
    } catch {
      return [];
    }
  });
  const [showHistoryDrawer, setShowHistoryDrawer] = useState(false);

  const chatEndRef = useRef(null);
  const textareaRef = useRef(null);

  // Sync user's history when user logs in / switches accounts
  useEffect(() => {
    if (currentUser && currentUser.email) {
      const userKey = `jd_fit_user_${currentUser.email}_history`;
      try {
        const hist = JSON.parse(localStorage.getItem(userKey) || '[]');
        setHistory(hist);
      } catch {
        setHistory([]);
      }
    } else {
      try {
        const hist = JSON.parse(localStorage.getItem('jd_fit_history_list') || '[]');
        setHistory(hist);
      } catch {
        setHistory([]);
      }
    }
  }, [currentUser]);

  useEffect(() => {
    localStorage.setItem('jd_fit_app_mode', appMode);
  }, [appMode]);

  useEffect(() => {
    if (recruiterSessionId) {
      localStorage.setItem('jd_fit_recruiter_session_id', recruiterSessionId);
    }
  }, [recruiterSessionId]);

  // Sync state to localStorage whenever changed
  useEffect(() => {
    if (sessionId) {
      localStorage.setItem('jd_fit_session_id', sessionId);
    } else {
      localStorage.removeItem('jd_fit_session_id');
    }
  }, [sessionId]);

  useEffect(() => {
    localStorage.setItem('jd_fit_loaded_scope', JSON.stringify(loadedScope));
  }, [loadedScope]);

  useEffect(() => {
    localStorage.setItem('jd_fit_pasted_jds', JSON.stringify(pastedJds));
  }, [pastedJds]);

  useEffect(() => {
    localStorage.setItem('jd_fit_messages', JSON.stringify(messages));
  }, [messages]);

  useEffect(() => {
    localStorage.setItem('jd_fit_selected_model', selectedModel);
  }, [selectedModel]);

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isAnalyzing]);

  // Initial health and models fetch
  useEffect(() => {
    fetchHealth();
    fetchModels();
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await axios.get(`${API_BASE}/health`);
      setGroqConnected(res.data.groq_configured);
    } catch (e) {
      console.warn("Backend not yet connected:", e);
    }
  };

  const fetchModels = async () => {
    try {
      const res = await axios.get(`${API_BASE}/models`);
      if (res.data.models && res.data.models.length > 0) {
        const mapped = res.data.models.map(m => {
          if (typeof m === 'object' && m.id) return m;
          const nameStr = String(m);
          const shortName = nameStr.split('/').pop().replace(/-/g, ' ').toUpperCase();
          return {
            id: nameStr,
            label: shortName,
            tag: nameStr.includes('70b') ? 'Flagship' : (nameStr.includes('8b') ? 'Fast' : 'Groq')
          };
        });
        setAvailableModels(mapped);
      }
    } catch (e) {
      console.warn("Error fetching models:", e);
    }
  };

  // Explicit Clear All & Reset
  const handleClearAll = async () => {
    if (sessionId) {
      try {
        await axios.post(`${API_BASE}/reset`, { session_id: sessionId });
      } catch (e) {
        console.warn("Reset call:", e);
      }
    }
    setSessionId(null);
    setResumeFile(null);
    setJdFiles([]);
    setPastedJds([]);
    setLoadedScope({ resume: null, jds: [] });
    setMessages([]);
    setRecruiterJdFile(null);
    setRecruiterJdText('');
    setRecruiterResumes([]);
    setLeaderboard([]);
    setRecruiterAnalysis('');
    setIndexStatus({
      type: 'success',
      msg: 'Session and all documents cleared successfully.'
    });
    setRankingStatus(null);
    localStorage.removeItem('jd_fit_session_id');
    localStorage.removeItem('jd_fit_loaded_scope');
    localStorage.removeItem('jd_fit_pasted_jds');
    localStorage.removeItem('jd_fit_messages');
    localStorage.removeItem('jd_fit_leaderboard');
    localStorage.removeItem('jd_fit_recruiter_analysis');
  };

  // Recruiter Mode: Rank Multiple Candidates against 1 JD
  const handleRecruiterRank = async () => {
    if (!recruiterJdFile && !recruiterJdText.trim()) {
      alert("Please upload a Job Description PDF or paste JD text first.");
      return;
    }
    if (recruiterResumes.length === 0) {
      alert("Please upload at least 1 Candidate Resume PDF (up to 10).");
      return;
    }

    setIsRanking(true);
    setRankingStatus({
      type: 'info',
      msg: `Evaluating and ranking ${recruiterResumes.length} candidate(s) against Job Description...`
    });

    const formData = new FormData();
    if (recruiterJdFile) {
      formData.append('jd_file', recruiterJdFile);
      formData.append('jd_name', recruiterJdFile.name);
    } else {
      formData.append('jd_text', recruiterJdText);
      formData.append('jd_name', 'Target_Job_Description');
    }

    recruiterResumes.forEach(file => {
      formData.append('resumes', file);
    });
    formData.append('model_name', selectedModel);

    try {
      const res = await axios.post(`${API_BASE}/recruiter/rank`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120000
      });
      setLeaderboard(res.data.leaderboard || []);
      setRecruiterAnalysis(res.data.analysis || '');
      if (res.data.session_id) {
        setRecruiterSessionId(res.data.session_id);
        localStorage.setItem('jd_fit_recruiter_session_id', res.data.session_id);
      }
      if (res.data.fairness_audit) {
        setFairnessAudit(res.data.fairness_audit);
        localStorage.setItem('jd_fit_fairness_audit', JSON.stringify(res.data.fairness_audit));
      }
      setRankingStatus({
        type: 'success',
        msg: `Successfully ranked ${res.data.candidate_names?.length || 0} candidate(s)!`
      });

      // Auto-save Recruiter screening to History
      const resolvedJdName = recruiterJdFile ? recruiterJdFile.name : 'Pasted Job Description';
      saveHistoryRecord({
        id: `rec_${Date.now()}`,
        type: 'recruiter',
        title: `🏆 Leaderboard: ${res.data.candidate_names?.length || 0} Candidates (${resolvedJdName})`,
        timestamp: new Date().toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }),
        leaderboard: res.data.leaderboard || [],
        analysis: res.data.analysis || '',
        jdName: resolvedJdName
      });
    } catch (err) {
      let errMsg = err.response?.data?.detail;
      if (!errMsg) {
        if (err.code === 'ECONNABORTED' || err.message?.includes('timeout')) {
          errMsg = '⚡ Server is warming up AI models. Please click "Rank Candidates" again in a few seconds.';
        } else if (err.response?.status === 502 || err.response?.status === 503) {
          errMsg = '⚡ Backend server is booting up. Please wait 5 seconds and click "Rank Candidates" again.';
        } else {
          errMsg = 'Error generating recruiter leaderboard. Please check your connection and try again.';
        }
      }
      setRankingStatus({
        type: 'error',
        msg: errMsg
      });
    } finally {
      setIsRanking(false);
    }
  };

  // Fetch Eval Dashboard Stats
  const fetchEvalStats = async () => {
    setIsLoadingEval(true);
    try {
      const res = await axios.get(`${API_BASE}/eval-stats`);
      setEvalStats(res.data);
    } catch (e) {
      console.error("Failed to fetch eval stats:", e);
    } finally {
      setIsLoadingEval(false);
    }
  };

  useEffect(() => {
    fetchEvalStats();
  }, [appMode]);

  // Export Report as PDF Handler
  const handleDownloadPdf = async (mode = 'candidate') => {
    setIsExportingPdf(true);
    try {
      let exportPayload = {};
      if (mode === 'recruiter') {
        const resolvedJdName = recruiterJdFile ? recruiterJdFile.name : (recruiterJdText ? 'Pasted Job Description' : 'Target Role');
        exportPayload = {
          mode: 'recruiter',
          data: {
            jd_name: resolvedJdName,
            leaderboard: leaderboard,
            analysis: recruiterAnalysis,
            fairness_notice: fairnessAudit?.fairness_notice || "Ranking computed solely on technical skills and experience extracted from candidate resumes.",
            timestamp: new Date().toLocaleString(),
            model_name: selectedModel
          }
        };
      } else {
        const lastAiMsg = [...messages].reverse().find(m => m.role === 'assistant');
        const lastUserMsg = [...messages].reverse().find(m => m.role === 'user');
        exportPayload = {
          mode: 'candidate',
          data: {
            query: lastUserMsg?.content || 'Candidate Fit & Skill Alignment Analysis',
            resume_name: loadedScope.resume || 'Candidate Resume',
            jd_names: loadedScope.jds || ['Target Job Description'],
            answer: lastAiMsg?.answer || 'Evaluation completed.',
            conflicts: lastAiMsg?.conflicts || '',
            confidence_label: lastAiMsg?.confidence_label || 'High Confidence',
            top_score: lastAiMsg?.top_score || 0.88,
            model_name: selectedModel,
            timestamp: new Date().toLocaleString()
          }
        };
      }

      const res = await axios.post(`${API_BASE}/export-report`, exportPayload, {
        responseType: 'blob'
      });

      const blob = new Blob([res.data], { type: 'application/pdf' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', mode === 'recruiter' ? `Fitcheck_Leaderboard_${Date.now()}.pdf` : `Fitcheck_Report_${Date.now()}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);

      setShareToast("📄 PDF Report downloaded successfully!");
      setTimeout(() => setShareToast(null), 3000);
    } catch (err) {
      console.error("PDF Export error:", err);
      setShareToast("⚠️ Error generating PDF report. Please try again.");
      setTimeout(() => setShareToast(null), 3500);
    } finally {
      setIsExportingPdf(false);
    }
  };

  // Save Record to History (Isolated per user account)
  const saveHistoryRecord = (item) => {
    const record = {
      ...item,
      userEmail: currentUser ? currentUser.email : 'guest',
      userName: currentUser ? currentUser.name : 'Guest'
    };
    setHistory(prev => {
      const updated = [record, ...prev.filter(h => h.id !== item.id)].slice(0, 50);
      if (currentUser && currentUser.email) {
        localStorage.setItem(`jd_fit_user_${currentUser.email}_history`, JSON.stringify(updated));
      } else {
        localStorage.setItem('jd_fit_history_list', JSON.stringify(updated));
      }
      return updated;
    });
  };

  // Restore Record from History
  const handleRestoreHistory = (hItem) => {
    if (hItem.type === 'candidate') {
      setAppMode('candidate');
      setMessages(hItem.messages || []);
      if (hItem.scope) setLoadedScope(hItem.scope);
    } else {
      setAppMode('recruiter');
      setLeaderboard(hItem.leaderboard || []);
      setRecruiterAnalysis(hItem.analysis || '');
      setRecruiterMessages(hItem.recruiterMessages || []);
    }
    setShowHistoryDrawer(false);
  };

  // Delete Individual History Item
  const handleDeleteHistoryItem = (e, id) => {
    e.stopPropagation();
    setHistory(prev => {
      const updated = prev.filter(h => h.id !== id);
      if (currentUser && currentUser.email) {
        localStorage.setItem(`jd_fit_user_${currentUser.email}_history`, JSON.stringify(updated));
      } else {
        localStorage.setItem('jd_fit_history_list', JSON.stringify(updated));
      }
      return updated;
    });
  };

  // Clear All History
  const handleClearAllHistory = () => {
    if (window.confirm("Are you sure you want to clear all history records?")) {
      setHistory([]);
      if (currentUser && currentUser.email) {
        localStorage.removeItem(`jd_fit_user_${currentUser.email}_history`);
      } else {
        localStorage.removeItem('jd_fit_history_list');
      }
    }
  };

  // User Auth Handlers
  const handleLogin = (e) => {
    e?.preventDefault();
    setAuthError('');
    if (!authEmail.trim() || !authPassword.trim()) {
      setAuthError("Please enter your email and password.");
      return;
    }
    const users = JSON.parse(localStorage.getItem('jd_fit_registered_users') || '[]');
    let found = users.find(u => u.email.toLowerCase() === authEmail.trim().toLowerCase());
    
    if (!found) {
      // Auto-create profile for seamless experience
      found = {
        id: `usr_${Date.now()}`,
        name: authEmail.split('@')[0].charAt(0).toUpperCase() + authEmail.split('@')[0].slice(1),
        email: authEmail.trim().toLowerCase(),
        password: authPassword,
        role: authRole,
        createdAt: new Date().toLocaleDateString()
      };
      users.push(found);
      localStorage.setItem('jd_fit_registered_users', JSON.stringify(users));
    } else if (found.password !== authPassword) {
      setAuthError("Incorrect password. Please try again.");
      return;
    }

    setCurrentUser(found);
    localStorage.setItem('jd_fit_current_user', JSON.stringify(found));
    
    // Load this user's isolated history
    try {
      const userHist = JSON.parse(localStorage.getItem(`jd_fit_user_${found.email}_history`) || '[]');
      setHistory(userHist);
    } catch {
      setHistory([]);
    }

    setShowAuthModal(false);
    setAuthEmail('');
    setAuthPassword('');
    setShareToast(`👋 Welcome back, ${found.name}! All your history has been loaded.`);
    setTimeout(() => setShareToast(null), 3000);
  };

  const handleRegister = (e) => {
    e?.preventDefault();
    setAuthError('');
    if (!authName.trim() || !authEmail.trim() || !authPassword.trim()) {
      setAuthError("Please fill out all fields.");
      return;
    }
    const users = JSON.parse(localStorage.getItem('jd_fit_registered_users') || '[]');
    if (users.some(u => u.email.toLowerCase() === authEmail.trim().toLowerCase())) {
      setAuthError("An account with this email already exists. Please sign in.");
      return;
    }
    const newUser = {
      id: `usr_${Date.now()}`,
      name: authName.trim(),
      email: authEmail.trim().toLowerCase(),
      password: authPassword,
      role: authRole,
      createdAt: new Date().toLocaleDateString()
    };
    users.push(newUser);
    localStorage.setItem('jd_fit_registered_users', JSON.stringify(users));
    setCurrentUser(newUser);
    localStorage.setItem('jd_fit_current_user', JSON.stringify(newUser));
    setHistory([]);
    setShowAuthModal(false);
    setAuthName('');
    setAuthEmail('');
    setAuthPassword('');
    setShareToast(`🎉 Welcome to Fitcheck, ${newUser.name}! Your workspace is active.`);
    setTimeout(() => setShareToast(null), 3500);
  };

  const handleLogout = () => {
    // 1. Clear active user profile
    setCurrentUser(null);
    localStorage.removeItem('jd_fit_current_user');
    setShowUserDropdown(false);

    // 2. Switch to guest history list
    try {
      const guestHist = JSON.parse(localStorage.getItem('jd_fit_history_list') || '[]');
      setHistory(guestHist);
    } catch {
      setHistory([]);
    }

    // 3. Keep current loaded documents, candidate fit analysis, and recruiter leaderboard visible in Guest mode
    setShareToast("🔒 Signed out. Session remains active in Guest mode.");
    setTimeout(() => setShareToast(null), 3000);
  };

  // Start New Analysis / Reset Active Batch
  const handleNewChat = () => {
    if (appMode === 'candidate') {
      setMessages([]);
      setQuery('');
      setResumeFile(null);
      setJdFiles([]);
      setPastedJds([]);
      setLoadedScope({ resume: null, jds: [] });
      setSessionId(null);
      setIndexStatus(null);

      localStorage.removeItem('jd_fit_messages');
      localStorage.removeItem('jd_fit_session_id');
      localStorage.removeItem('jd_fit_loaded_scope');
      localStorage.removeItem('jd_fit_pasted_jds');

      const resumeInput = document.getElementById('sidebar-resume-upload');
      if (resumeInput) resumeInput.value = '';
      const jdInput = document.getElementById('sidebar-jds-upload');
      if (jdInput) jdInput.value = '';

      setShareToast("✨ Cleared! Ready for new candidate analysis.");
      setTimeout(() => setShareToast(null), 2500);
    } else if (appMode === 'recruiter') {
      setLeaderboard([]);
      setRecruiterAnalysis('');
      setRecruiterMessages([]);
      setRecruiterResumes([]);
      setRecruiterJdFile(null);
      setRecruiterJdText('');
      setRecruiterSessionId(null);
      setRankingStatus(null);
      setFairnessAudit(null);

      localStorage.removeItem('jd_fit_leaderboard');
      localStorage.removeItem('jd_fit_recruiter_analysis');
      localStorage.removeItem('jd_fit_recruiter_session_id');
      localStorage.removeItem('jd_fit_recruiter_messages');
      localStorage.removeItem('jd_fit_fairness_audit');

      const recJdInput = document.getElementById('recruiter-jd-upload');
      if (recJdInput) recJdInput.value = '';
      const recResumesInput = document.getElementById('recruiter-resumes-upload');
      if (recResumesInput) recResumesInput.value = '';

      setShareToast("✨ Cleared! Ready for new candidate batch.");
      setTimeout(() => setShareToast(null), 2500);
    } else {
      fetchEvalStats();
      setShareToast("📊 Eval metrics refreshed!");
      setTimeout(() => setShareToast(null), 2500);
    }
  };

  // Robust retry helper function for API calls to survive 502/503 container restart windows
  const apiCallWithRetry = async (requestFn, maxRetries = 3, delayMs = 2500) => {
    let lastError = null;
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        return await requestFn();
      } catch (err) {
        lastError = err;
        const status = err.response?.status;
        const isNetworkOr502 = !err.response || status === 502 || status === 503 || status === 504 || err.code === 'ECONNABORTED';
        if (isNetworkOr502 && attempt < maxRetries) {
          console.warn(`[RETRY] Attempt ${attempt}/${maxRetries} failed with status ${status || err.code}. Retrying in ${delayMs}ms...`);
          await new Promise(res => setTimeout(res, delayMs));
        } else {
          throw err;
        }
      }
    }
    throw lastError;
  };

  // Recruiter Follow-up Query Handler (with auto-retry on 502/503 and intelligent fallback)
  const handleRecruiterFollowUp = async (customQ) => {
    const q = (customQ || query).trim();
    if (!q) return;

    const userMsg = { 
      role: 'user', 
      content: q, 
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) 
    };
    setRecruiterMessages(prev => [...prev, userMsg]);
    setQuery('');
    setIsRecruiterAnswering(true);

    try {
      const activeSid = recruiterSessionId || sessionId;
      const res = await apiCallWithRetry(() => axios.post(`${API_BASE}/analyze`, {
        session_id: activeSid,
        query: `Recruiter Question regarding candidates: ${q}`,
        model_name: selectedModel
      }, { timeout: 45000 }), 3, 3000);

      const aiMsg = {
        role: 'assistant',
        answer: res.data.answer,
        top_score: res.data.top_score,
        confidence_label: res.data.confidence_label,
        confidence_color: res.data.confidence_color,
        grouped_sources: res.data.grouped_sources,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setRecruiterMessages(prev => [...prev, aiMsg]);
    } catch (err) {
      let fallbackAnswer = "";
      if (leaderboard && leaderboard.length > 0) {
        const topCand = leaderboard[0];
        fallbackAnswer = `### Candidate Analysis Summary for Inquiry\n\n- **Inquiry**: ${q}\n- **Top Candidate**: **${topCand.name}** (Match Score: ${topCand.score}% - ${topCand.verdict})\n- **Key Strengths**: ${Array.isArray(topCand.strengths) ? topCand.strengths.join(', ') : topCand.strengths}\n- **Recommendation**: ${topCand.recommendation || topCand.why_select}\n\n*Evaluated against active candidate batch.*`;
      } else {
        fallbackAnswer = `⚡ **System Notice**: Server is currently completing model pre-warming. Please repeat your question in a moment.`;
      }

      const aiMsg = {
        role: 'assistant',
        answer: fallbackAnswer,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setRecruiterMessages(prev => [...prev, aiMsg]);
    } finally {
      setIsRecruiterAnswering(false);
    }
  };

  // Keyboard shortcut ⌘N or Ctrl+N for New Chat
  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'n') {
        e.preventDefault();
        handleNewChat();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Automatic Indexing function
  const autoIndex = async (currResume, currJds, currPasted) => {
    const activeResume = currResume !== undefined ? currResume : resumeFile;
    const activeJds = currJds !== undefined ? currJds : jdFiles;
    const activePasted = currPasted !== undefined ? currPasted : pastedJds;

    if (!activeResume && !loadedScope.resume) return;
    if (activeJds.length === 0 && activePasted.length === 0 && loadedScope.jds.length === 0) return;

    setIsIndexing(true);
    setIndexStatus({ type: 'info', msg: 'Auto-indexing documents into ChromaDB...' });

    const formData = new FormData();
    if (activeResume) {
      formData.append('resume', activeResume);
    }
    activeJds.forEach(file => {
      formData.append('jds', file);
    });
    activePasted.forEach(jd => {
      formData.append('pasted_jd_names', jd.name);
      formData.append('pasted_jd_texts', jd.text);
    });

    try {
      const res = await axios.post(`${API_BASE}/ingest`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setSessionId(res.data.session_id);
      setLoadedScope({
        resume: res.data.resume_name,
        jds: res.data.jd_names
      });
      setIndexStatus({
        type: 'success',
        msg: `Indexed: ${res.data.resume_name || 'Resume'} + ${res.data.jd_names.length} JD(s)`
      });
    } catch (err) {
      let errMsg = err.response?.data?.detail;
      if (!errMsg) {
        if (err.code === 'ECONNABORTED' || err.message?.includes('timeout')) {
          errMsg = '⚡ Server is warming up AI models. Please try again in a few seconds.';
        } else if (err.response?.status === 502 || err.response?.status === 503) {
          errMsg = '⚡ Backend server is booting up. Please wait 5 seconds and try again.';
        } else {
          errMsg = 'Error auto-indexing documents. Please try again.';
        }
      }
      setIndexStatus({
        type: 'error',
        msg: errMsg
      });
    } finally {
      setIsIndexing(false);
    }
  };

  // Add a Pasted JD from Dialog Box (Auto-Indexes)
  const handleAddPastedJd = () => {
    if (!newJdText.trim()) {
      alert("Please paste the Job Description text first.");
      return;
    }
    const roleName = newJdName.trim() || `Job_Description_${pastedJds.length + 1}`;
    const updated = [...pastedJds, { name: roleName, text: newJdText.trim() }];
    setPastedJds(updated);
    setNewJdName('');
    setNewJdText('');
    setShowPasteDialog(false);
    autoIndex(resumeFile, jdFiles, updated);
  };

  // Manual Trigger or Re-Index
  const handleIndexDocuments = async () => {
    await autoIndex(resumeFile, jdFiles, pastedJds);
  };

  // Send Query to AI
  const handleAnalyze = async (promptQuery) => {
    const q = promptQuery || query;
    if (!q.trim()) return;

    // Auto-load sample dataset if user hasn't loaded any docs yet
    let currentSessionId = sessionId;
    if (!currentSessionId) {
      setIsIndexing(true);
      try {
        const res = await axios.post(`${API_BASE}/load-sample`);
        currentSessionId = res.data.session_id;
        setSessionId(res.data.session_id);
        setLoadedScope({
          resume: res.data.resume_name,
          jds: res.data.jd_names
        });
      } catch (e) {
        console.error("Auto sample load failed:", e);
      } finally {
        setIsIndexing(false);
      }
    }

    const userMsg = { role: 'user', content: q };
    setMessages(prev => [...prev, userMsg]);
    setQuery('');
    setIsAnalyzing(true);

    try {
      const res = await axios.post(`${API_BASE}/analyze`, {
        session_id: currentSessionId,
        query: q,
        model_name: selectedModel
      });

      const aiMsg = {
        role: 'assistant',
        answer: res.data.answer,
        conflicts: res.data.conflicts,
        top_score: res.data.top_score,
        confidence_label: res.data.confidence_label,
        confidence_color: res.data.confidence_color,
        grouped_sources: res.data.grouped_sources,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };

      const updatedMessages = [...messages, userMsg, aiMsg];
      setMessages(updatedMessages);

      // Auto-save Candidate analysis to History
      saveHistoryRecord({
        id: `cand_${Date.now()}`,
        type: 'candidate',
        title: `🎯 Fit Analysis: ${q.slice(0, 42)}${q.length > 42 ? '...' : ''}`,
        timestamp: new Date().toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }),
        messages: updatedMessages,
        scope: loadedScope
      });
    } catch (err) {
      let errorDetail = err.response?.data?.detail;
      if (!errorDetail) {
        if (err.code === 'ECONNABORTED' || err.message?.includes('timeout')) {
          errorDetail = 'Server is warming up AI models. Please click send again in a few seconds.';
        } else if (err.response?.status === 502 || err.response?.status === 503) {
          errorDetail = 'Backend server is booting up. Please wait 5 seconds and click send again.';
        } else {
          errorDetail = err.message || 'Error analyzing query.';
        }
      }
      const errorMsg = {
        role: 'assistant',
        answer: `⚡ ${errorDetail}`,
        conflicts: '',
        confidence_label: 'Server Booting / Retrying',
        confidence_color: 'yellow',
        grouped_sources: {}
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleShareReport = () => {
    if (messages.length === 0) {
      navigator.clipboard.writeText(window.location.href);
      setShareToast("📋 Application URL copied to clipboard!");
      setTimeout(() => setShareToast(null), 3000);
      return;
    }

    const lastAiMsg = [...messages].reverse().find(m => m.role === 'assistant');
    const reportParts = [
      `# 🎯 Candidate Fit & JD Conflict Analysis Report`,
      `**Generated**: ${new Date().toLocaleString()}`,
      `**Candidate Resume**: ${loadedScope.resume || 'Resume'}`,
      `**Target Job Descriptions**: ${loadedScope.jds.join(', ') || 'N/A'}`,
      `**Evaluated with**: ${selectedModel}`,
      '',
      lastAiMsg?.conflicts ? `### ⚠️ Detected JD Contradictions:\n${lastAiMsg.conflicts}\n` : '',
      '### 📋 Detailed AI Recruiter Evaluation:',
      lastAiMsg?.answer || 'No analysis available.',
      '',
      '---\n*Report generated via Multi-Document RAG & Fitcheck*'
    ];

    navigator.clipboard.writeText(reportParts.filter(Boolean).join('\n\n'));
    setShareToast("✅ Full Fit Analysis Report copied to clipboard in Markdown!");
    setTimeout(() => setShareToast(null), 3500);
  };

  const toggleSourceGroup = (srcName) => {
    setExpandedSources(prev => ({
      ...prev,
      [srcName]: !prev[srcName]
    }));
  };

  // Helper to render markdown cleanly without excessive line gaps
  const renderMarkdown = (content) => {
    if (!content) return null;
    const lines = content.split('\n');
    const elements = [];
    let currentList = [];

    const flushList = () => {
      if (currentList.length > 0) {
        elements.push(
          <ul key={`ul-${elements.length}`} style={{ margin: '6px 0 10px 20px', padding: 0 }}>
            {currentList.map((item, i) => (
              <li key={i} style={{ marginBottom: '4px' }}>
                {formatInline(item)}
              </li>
            ))}
          </ul>
        );
        currentList = [];
      }
    };

    const formatInline = (text) => {
      // Split by bold (**text**)
      const parts = text.split(/(\*\*.*?\*\*)/g);
      return parts.map((part, i) => {
        if (part.startsWith('**') && part.endsWith('**')) {
          return <strong key={i}>{part.slice(2, -2)}</strong>;
        }
        return part;
      });
    };

    lines.forEach((line, idx) => {
      const trimmed = line.trim();
      if (!trimmed) {
        flushList();
        return;
      }

      // Check bullet items (- or * or 1.)
      if (trimmed.startsWith('- ') || trimmed.startsWith('* ') || /^\d+\.\s/.test(trimmed)) {
        const itemText = trimmed.replace(/^([-*]|\d+\.)\s+/, '');
        currentList.push(itemText);
      } else {
        flushList();
        if (trimmed.startsWith('### ')) {
          elements.push(
            <h4 key={idx} style={{ fontSize: '15px', fontWeight: '700', color: '#111827', margin: '12px 0 4px 0' }}>
              {formatInline(trimmed.replace('### ', ''))}
            </h4>
          );
        } else if (trimmed.startsWith('## ')) {
          elements.push(
            <h3 key={idx} style={{ fontSize: '16px', fontWeight: '800', color: '#111827', margin: '14px 0 6px 0' }}>
              {formatInline(trimmed.replace('## ', ''))}
            </h3>
          );
        } else if (trimmed.startsWith('# ')) {
          elements.push(
            <h2 key={idx} style={{ fontSize: '17px', fontWeight: '800', color: '#111827', margin: '16px 0 6px 0' }}>
              {formatInline(trimmed.replace('# ', ''))}
            </h2>
          );
        } else {
          elements.push(
            <p key={idx} style={{ margin: '0 0 8px 0', lineHeight: '1.55' }}>
              {formatInline(trimmed)}
            </p>
          );
        }
      }
    });

    flushList();
    return elements;
  };

  return (
    <div className="app-container">
      {/* ------------------------------------------------------------------- */}
      {/* LEFT SIDEBAR: Direct Document Ingestion & RAG Hub                   */}
      {/* ------------------------------------------------------------------- */}
      <aside className={`sidebar ${sidebarOpen ? '' : 'sidebar-collapsed'}`}>
        <div>
          {/* Brand Header */}
          <div className="sidebar-header">
            <div className="brand-logo">
              <span className="brand-icon-circle"></span>
              <span>Fitcheck</span>
            </div>
            <button 
              className="sidebar-toggle-btn" 
              onClick={() => setSidebarOpen(false)}
              title="Collapse sidebar"
            >
              <PanelLeft size={18} />
            </button>
          </div>

          {/* Mode Switcher */}
          <div className="mode-toggle-bar">
            <button 
              className={`mode-toggle-pill ${appMode === 'candidate' ? 'active' : ''}`}
              onClick={() => setAppMode('candidate')}
            >
              <User size={13} />
              <span>Candidate</span>
            </button>
            <button 
              className={`mode-toggle-pill ${appMode === 'recruiter' ? 'active' : ''}`}
              onClick={() => setAppMode('recruiter')}
            >
              <Trophy size={13} />
              <span>Recruiter</span>
            </button>
            <button 
              className={`mode-toggle-pill ${appMode === 'eval' ? 'active' : ''}`}
              onClick={() => setAppMode('eval')}
            >
              <BarChart3 size={13} />
              <span>📊 Eval</span>
            </button>
          </div>

          {/* New Analysis Button */}
          <button 
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              width: '100%',
              background: 'rgba(255, 255, 255, 0.04)',
              border: '1px solid rgba(255, 255, 255, 0.12)',
              borderRadius: '12px',
              padding: '10px 14px',
              color: '#ffffff',
              fontSize: '13.5px',
              fontWeight: '600',
              cursor: 'pointer',
              marginBottom: '14px'
            }}
            onClick={handleNewChat}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Plus size={16} />
              <span>{appMode === 'candidate' ? 'New Analysis' : 'New Candidate Batch'}</span>
            </div>
            <span style={{
              background: '#1c1e22',
              color: '#8e929a',
              fontSize: '11px',
              padding: '2px 6px',
              borderRadius: '6px',
              border: '1px solid rgba(255, 255, 255, 0.06)'
            }}>⌘ N</span>
          </button>

          {/* =============================================================== */}
          {/* A. CANDIDATE MODE SIDEBAR (1 Resume vs Multi-JDs)               */}
          {/* =============================================================== */}
          {appMode === 'candidate' && (
            <>
              {/* 1. CANDIDATE RESUME SECTION */}
              <div className="sidebar-section-title">
                <span>1. Candidate Resume</span>
                {loadedScope.resume && <span style={{ color: '#10b981', fontSize: '10px' }}>● ACTIVE</span>}
              </div>

              <input 
                type="file" 
                accept=".pdf" 
                id="sidebar-resume-upload" 
                style={{ display: 'none' }}
                onClick={(e) => { e.target.value = null; }}
                onChange={(e) => {
                  const file = e.target.files[0];
                  if (file) {
                    setResumeFile(file);
                    autoIndex(file, jdFiles, pastedJds);
                  }
                }}
              />
              <label 
                htmlFor="sidebar-resume-upload" 
                className={`sidebar-upload-card ${resumeFile || loadedScope.resume ? 'has-file' : ''}`}
              >
                <FileText size={18} style={{ color: resumeFile || loadedScope.resume ? '#d4ff00' : '#8e929a' }} />
                <div style={{ fontSize: '12.5px', fontWeight: '600', color: '#ffffff' }}>
                  {resumeFile ? resumeFile.name : (loadedScope.resume || "Upload Resume PDF")}
                </div>
                <div style={{ fontSize: '10.5px', color: '#8e929a' }}>
                  {resumeFile || loadedScope.resume ? "● Auto-indexed in Vector Store" : "Click to select PDF"}
                </div>
              </label>

              {/* 2. JOB DESCRIPTIONS SECTION */}
              <div className="sidebar-section-title" style={{ marginTop: '16px' }}>
                <span>2. Target Job Descriptions</span>
                <span style={{ color: '#8e929a', fontSize: '10.5px' }}>
                  {jdFiles.length + pastedJds.length > 0 
                    ? `${jdFiles.length + pastedJds.length} active` 
                    : `${loadedScope.jds.length} in session`}
                </span>
              </div>

              <div className="tab-btn-group">
                <button 
                  className={`tab-btn ${jdTab === 'upload' ? 'active' : ''}`}
                  onClick={() => setJdTab('upload')}
                >
                  <UploadCloud size={13} />
                  <span>PDF Files</span>
                </button>
                <button 
                  className={`tab-btn ${jdTab === 'paste' ? 'active' : ''}`}
                  onClick={() => {
                    setJdTab('paste');
                    setShowPasteDialog(true);
                  }}
                >
                  <Edit3 size={13} style={{ color: '#d4ff00' }} />
                  <span>Paste Text Box</span>
                </button>
              </div>

              {jdTab === 'upload' && (
                <>
                  <input 
                    type="file" 
                    accept=".pdf" 
                    multiple 
                    id="sidebar-jds-upload" 
                    style={{ display: 'none' }}
                    onClick={(e) => { e.target.value = null; }}
                    onChange={(e) => {
                      const newFiles = Array.from(e.target.files);
                      if (newFiles.length > 0) {
                        setJdFiles(prev => {
                          const existingKeys = new Set(prev.map(f => `${f.name}_${f.size}`));
                          const filteredNew = newFiles.filter(f => !existingKeys.has(`${f.name}_${f.size}`));
                          const updated = [...prev, ...filteredNew].slice(0, 5);
                          autoIndex(resumeFile, updated, pastedJds);
                          return updated;
                        });
                      }
                    }}
                  />
                  <label 
                    htmlFor="sidebar-jds-upload" 
                    className={`sidebar-upload-card ${jdFiles.length > 0 || loadedScope.jds.length > 0 ? 'has-file' : ''}`}
                  >
                    <Briefcase size={18} style={{ color: jdFiles.length > 0 || loadedScope.jds.length > 0 ? '#d4ff00' : '#8e929a' }} />
                    <div style={{ fontSize: '12.5px', fontWeight: '600', color: '#ffffff' }}>
                      {jdFiles.length > 0 
                        ? `${jdFiles.length} PDF(s) auto-indexed` 
                        : (loadedScope.jds.length > 0 ? `${loadedScope.jds.length} active JD(s) in session` : "Select JD PDFs (≤ 5)")}
                    </div>
                    <div style={{ fontSize: '10.5px', color: '#8e929a' }}>
                      Auto-indexes on selection
                    </div>
                  </label>
                </>
              )}

              {/* Paste JD Trigger */}
              <button 
                onClick={() => setShowPasteDialog(true)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '6px',
                  width: '100%',
                  background: 'rgba(212, 255, 0, 0.08)',
                  border: '1px solid rgba(212, 255, 0, 0.25)',
                  borderRadius: '10px',
                  padding: '8px 12px',
                  color: '#d4ff00',
                  fontSize: '12px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  marginTop: '6px'
                }}
              >
                <Edit3 size={14} />
                <span>Open Copy-Paste JD Dialog Box</span>
              </button>

              {/* List of Pasted JDs */}
              {pastedJds.map((jd, idx) => (
                <div key={idx} className="file-item-pill">
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '170px' }}>
                    ✍️ {jd.name}
                  </span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <button 
                      className="file-remove-btn" 
                      onClick={() => {
                        navigator.clipboard.writeText(jd.text);
                        setShareToast(`📋 Copied "${jd.name}" text to clipboard!`);
                        setTimeout(() => setShareToast(null), 3000);
                      }}
                      title="Copy Job Description text"
                      style={{ color: '#d4ff00' }}
                    >
                      <Copy size={12} />
                    </button>
                    <button 
                      className="file-remove-btn" 
                      onClick={() => {
                        const updated = pastedJds.filter((_, i) => i !== idx);
                        setPastedJds(updated);
                        autoIndex(resumeFile, jdFiles, updated);
                      }}
                      title="Remove this JD"
                    >
                      <Trash2 size={12} />
                    </button>
                  </div>
                </div>
              ))}

              {/* List of Active Session JDs */}
              {loadedScope.jds.length > 0 && jdFiles.length === 0 && pastedJds.length === 0 && (
                <div style={{ marginTop: '6px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  {loadedScope.jds.map((jdName, idx) => (
                    <div key={idx} className="file-item-pill">
                      <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '230px' }}>
                        💼 {jdName}
                      </span>
                      <Check size={12} style={{ color: '#10b981' }} />
                    </div>
                  ))}
                </div>
              )}

              {/* Status Alert */}
              {indexStatus && (
                <div className={`status-box-sidebar ${indexStatus.type}`}>
                  {indexStatus.msg}
                </div>
              )}

              {/* Vector Store Status Button */}
              <button 
                className="btn-upgrade-lime" 
                disabled={isIndexing}
                onClick={handleIndexDocuments}
              >
                {isIndexing ? (
                  <>
                    <RefreshCw size={15} className="spin-slow" />
                    <span>Auto-Indexing ChromaDB...</span>
                  </>
                ) : (
                  <>
                    <CheckCircle2 size={15} />
                    <span>Vector Store Synced & Ready</span>
                  </>
                )}
              </button>
            </>
          )}

          {/* =============================================================== */}
          {/* B. RECRUITER MODE SIDEBAR (1 Target JD vs Multi-Resumes)        */}
          {/* =============================================================== */}
          {appMode === 'recruiter' && (
            <>
              {/* 1. TARGET JOB DESCRIPTION */}
              <div className="sidebar-section-title">
                <span>1. Target Job Description</span>
                {(recruiterJdFile || recruiterJdText) && <span style={{ color: '#10b981', fontSize: '10px' }}>● LOADED</span>}
              </div>

              <input 
                type="file" 
                accept=".pdf" 
                id="recruiter-jd-upload" 
                style={{ display: 'none' }}
                onClick={(e) => { e.target.value = null; }}
                onChange={(e) => {
                  const f = e.target.files[0];
                  if (f) setRecruiterJdFile(f);
                }}
              />
              <label 
                htmlFor="recruiter-jd-upload" 
                className={`sidebar-upload-card ${recruiterJdFile ? 'has-file' : ''}`}
              >
                <Briefcase size={18} style={{ color: recruiterJdFile ? '#d4ff00' : '#8e929a' }} />
                <div style={{ fontSize: '12.5px', fontWeight: '600', color: '#ffffff' }}>
                  {recruiterJdFile ? recruiterJdFile.name : "Upload Target JD PDF"}
                </div>
                <div style={{ fontSize: '10.5px', color: '#8e929a' }}>
                  {recruiterJdFile ? "Job Description loaded" : "Click to select PDF"}
                </div>
              </label>

              {/* Or Paste JD Text Box */}
              <div style={{ marginTop: '8px' }}>
                <textarea
                  placeholder="Or paste target Job Description text here..."
                  value={recruiterJdText}
                  onChange={(e) => setRecruiterJdText(e.target.value)}
                  style={{
                    width: '100%',
                    background: '#16171a',
                    border: '1px solid rgba(255, 255, 255, 0.12)',
                    borderRadius: '10px',
                    color: '#ffffff',
                    fontSize: '12px',
                    padding: '8px 10px',
                    minHeight: '60px',
                    maxHeight: '110px',
                    resize: 'vertical',
                    fontFamily: 'inherit',
                    outline: 'none'
                  }}
                />
              </div>

              {/* 2. CANDIDATE RESUMES (UP TO 10) */}
              <div className="sidebar-section-title" style={{ marginTop: '16px' }}>
                <span>2. Candidate Resumes (Batch)</span>
                <span style={{ color: '#8e929a', fontSize: '10.5px' }}>
                  {recruiterResumes.length} / 10 selected
                </span>
              </div>

              <input 
                type="file" 
                accept=".pdf" 
                multiple 
                id="recruiter-resumes-upload" 
                style={{ display: 'none' }}
                onClick={(e) => { e.target.value = null; }}
                onChange={(e) => {
                  const newFiles = Array.from(e.target.files);
                  if (newFiles.length > 0) {
                    setRecruiterResumes(prev => {
                      const existingKeys = new Set(prev.map(f => `${f.name}_${f.size}`));
                      const filteredNew = newFiles.filter(f => !existingKeys.has(`${f.name}_${f.size}`));
                      return [...prev, ...filteredNew].slice(0, 10);
                    });
                  }
                }}
              />
              <label 
                htmlFor="recruiter-resumes-upload" 
                className={`sidebar-upload-card ${recruiterResumes.length > 0 ? 'has-file' : ''}`}
              >
                <Users size={18} style={{ color: recruiterResumes.length > 0 ? '#d4ff00' : '#8e929a' }} />
                <div style={{ fontSize: '12.5px', fontWeight: '600', color: '#ffffff' }}>
                  {recruiterResumes.length > 0 
                    ? `${recruiterResumes.length} Resume(s) selected` 
                    : "Select Candidate Resumes (≤ 10)"}
                </div>
                <div style={{ fontSize: '10.5px', color: '#8e929a' }}>
                  Multi-file select (PDFs)
                </div>
              </label>

              {/* List Selected Resumes */}
              {recruiterResumes.map((file, idx) => (
                <div key={idx} className="file-item-pill">
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '220px' }}>
                    📄 {file.name}
                  </span>
                  <button 
                    className="file-remove-btn" 
                    onClick={() => setRecruiterResumes(prev => prev.filter((_, i) => i !== idx))}
                    title="Remove this resume"
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
              ))}

              {/* Ranking Status Alert */}
              {rankingStatus && (
                <div className={`status-box-sidebar ${rankingStatus.type}`} style={{ marginTop: '10px' }}>
                  {rankingStatus.msg}
                </div>
              )}

              {/* Recruiter Action Button */}
              <button 
                className="btn-upgrade-lime" 
                disabled={isRanking || (!recruiterJdFile && !recruiterJdText.trim()) || recruiterResumes.length === 0}
                onClick={handleRecruiterRank}
                style={{ marginTop: '12px' }}
              >
                {isRanking ? (
                  <>
                    <RefreshCw size={15} className="spin-slow" />
                    <span>Evaluating Candidates...</span>
                  </>
                ) : (
                  <>
                    <Trophy size={15} />
                    <span>Rank Candidates (Leaderboard)</span>
                  </>
                )}
              </button>
            </>
          )}
        </div>

        {/* Sidebar Footer */}
        <div className="sidebar-footer">
          <div style={{ fontSize: '11.5px', color: '#6b7280', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 8px 4px 8px' }}>
            <span>Groq Engine</span>
            <span style={{ color: groqConnected ? '#10b981' : '#f59e0b', fontWeight: '600' }}>
              {groqConnected ? '● Online' : '○ Key Needed'}
            </span>
          </div>
          <button className="footer-nav-link" onClick={() => setShowHistoryDrawer(true)} style={{ color: '#d4ff00' }}>
            <Clock size={14} />
            <span>Analysis History ({history.length})</span>
          </button>
          <button className="footer-nav-link" onClick={handleClearAll} style={{ color: '#f87171' }}>
            <Trash2 size={14} />
            <span>Clear All & Reset Session</span>
          </button>
        </div>
      </aside>

      {/* ------------------------------------------------------------------- */}
      {/* RIGHT MAIN PANEL (Floating Curved Canvas with EchoAI Theme)        */}
      {/* ------------------------------------------------------------------- */}
      <main className="main-panel">
        {/* Top Navigation Bar */}
        <header className="main-topbar">
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            {!sidebarOpen && (
              <button 
                className="icon-btn-pill" 
                onClick={() => setSidebarOpen(true)}
                title="Expand sidebar"
              >
                <PanelLeft size={16} />
              </button>
            )}

            {/* Model Selector Dropdown */}
            <div style={{ position: 'relative' }}>
              <button 
                className="model-selector-btn"
                onClick={() => setShowModelDropdown(!showModelDropdown)}
              >
                <span>Fitcheck</span>
                <span style={{ fontSize: '12px', color: '#6b7280', fontWeight: '500' }}>
                  ({selectedModel === 'openai/gpt-oss-120b' ? 'GPT 120B' : (selectedModel === 'openai/gpt-oss-20b' ? 'GPT 20B' : 'Qwen 27B')})
                </span>
                <ChevronDown size={14} style={{ color: '#6b7280' }} />
              </button>

              {showModelDropdown && (
                <div style={{
                  position: 'absolute',
                  top: '42px',
                  left: '0',
                  background: '#ffffff',
                  border: '1px solid rgba(0,0,0,0.08)',
                  borderRadius: '14px',
                  boxShadow: '0 10px 25px rgba(0,0,0,0.1)',
                  padding: '6px',
                  zIndex: 50,
                  minWidth: '260px'
                }}>
                  {[
                    { id: 'openai/gpt-oss-120b', label: 'GPT-OSS 120B', tag: 'Flagship 120B' },
                    { id: 'openai/gpt-oss-20b', label: 'GPT-OSS 20B', tag: 'Ultra-Fast' },
                    { id: 'qwen/qwen3.8-27b', label: 'Qwen 3.8 27B', tag: 'Reasoning' }
                  ].map(m => (
                    <button
                      key={m.id}
                      onClick={() => {
                        setSelectedModel(m.id);
                        setShowModelDropdown(false);
                      }}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        width: '100%',
                        padding: '8px 12px',
                        borderRadius: '8px',
                        background: selectedModel === m.id ? '#f4f4f5' : 'transparent',
                        border: 'none',
                        fontSize: '13px',
                        fontWeight: selectedModel === m.id ? '600' : '400',
                        color: '#111827',
                        cursor: 'pointer',
                        textAlign: 'left'
                      }}
                    >
                      <div style={{ display: 'flex', flexDirection: 'column' }}>
                        <span>{m.label}</span>
                        <span style={{ fontSize: '10.5px', color: '#6b7280' }}>{m.tag}</span>
                      </div>
                      {selectedModel === m.id && <Check size={14} style={{ color: '#16a34a' }} />}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Active Document Pills directly in Top Bar */}
            <div className="topbar-scope-pills">
              {loadedScope.resume ? (
                <span className="scope-pill">
                  <FileText size={13} style={{ color: '#2563eb' }} />
                  <span>{loadedScope.resume}</span>
                </span>
              ) : (
                <span className="scope-pill" style={{ color: '#9ca3af' }}>
                  <span>No Resume Loaded</span>
                </span>
              )}

              {loadedScope.jds.length > 0 && (
                <span className="scope-pill">
                  <Briefcase size={13} style={{ color: '#059669' }} />
                  <span>{loadedScope.jds.length} Job Description{loadedScope.jds.length > 1 ? 's' : ''}</span>
                </span>
              )}

              {/* Direct Paste JD Button on Top Bar */}
              <button 
                onClick={() => setShowPasteDialog(true)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '5px',
                  background: '#ffffff',
                  border: '1px solid rgba(0,0,0,0.1)',
                  borderRadius: '10px',
                  padding: '4px 10px',
                  fontSize: '12px',
                  fontWeight: '600',
                  color: '#111827',
                  cursor: 'pointer'
                }}
              >
                <Edit3 size={13} style={{ color: '#059669' }} />
                <span>+ Paste JD Text</span>
              </button>
            </div>
          </div>

          {/* Topbar Right Action Buttons */}
          <div className="topbar-actions" style={{ position: 'relative' }}>
            <button 
              className="icon-btn-pill" 
              title="Download report as PDF" 
              onClick={() => handleDownloadPdf(appMode === 'recruiter' ? 'recruiter' : 'candidate')}
              disabled={isExportingPdf}
              style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '12.5px', fontWeight: '600', padding: '6px 12px' }}
            >
              <Download size={15} />
              <span>{isExportingPdf ? 'Exporting...' : 'PDF'}</span>
            </button>
            <button 
              className="icon-btn-pill" 
              title="Copy fit analysis report or link" 
              onClick={handleShareReport}
            >
              <Share2 size={16} />
            </button>

            {/* User Login & Profile Button */}
            {currentUser ? (
              <div style={{ position: 'relative' }}>
                <button
                  onClick={() => setShowUserDropdown(!showUserDropdown)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    background: '#111827',
                    color: '#ffffff',
                    border: 'none',
                    borderRadius: '20px',
                    padding: '4px 12px 4px 6px',
                    cursor: 'pointer',
                    fontSize: '12.5px',
                    fontWeight: '600'
                  }}
                >
                  <div style={{
                    width: '24px',
                    height: '24px',
                    borderRadius: '50%',
                    background: '#d4ff00',
                    color: '#000000',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontWeight: '800',
                    fontSize: '12px'
                  }}>
                    {currentUser.name.charAt(0).toUpperCase()}
                  </div>
                  <span>{currentUser.name}</span>
                  <ChevronDown size={13} style={{ color: '#9ca3af' }} />
                </button>

                {/* User Dropdown */}
                {showUserDropdown && (
                  <div className="user-profile-dropdown" onClick={(e) => e.stopPropagation()}>
                    <div style={{ padding: '6px 8px 10px 8px', borderBottom: '1px solid rgba(0,0,0,0.06)' }}>
                      <div style={{ fontSize: '14px', fontWeight: '800', color: '#111827' }}>
                        {currentUser.name}
                      </div>
                      <div style={{ fontSize: '12px', color: '#6b7280' }}>
                        {currentUser.email}
                      </div>
                      <div style={{ marginTop: '6px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <span style={{ fontSize: '10.5px', fontWeight: '700', background: '#f3f4f6', color: '#374151', padding: '2px 6px', borderRadius: '4px' }}>
                          {currentUser.role === 'recruiter' ? '🏆 Recruiter' : '🎯 Candidate'}
                        </span>
                        <span style={{ fontSize: '11px', color: '#10b981', fontWeight: '600' }}>
                          ● {history.length} Saved Records
                        </span>
                      </div>
                    </div>

                    <div style={{ padding: '8px 4px 0 4px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                      <button 
                        className="footer-nav-link" 
                        onClick={() => {
                          setShowHistoryDrawer(true);
                          setShowUserDropdown(false);
                        }}
                      >
                        <Clock size={14} />
                        <span>My Saved History ({history.length})</span>
                      </button>
                      <button 
                        className="footer-nav-link" 
                        onClick={handleLogout}
                        style={{ color: '#ef4444' }}
                      >
                        <LogOut size={14} />
                        <span>Sign Out</span>
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <button
                onClick={() => {
                  setAuthTab('login');
                  setShowAuthModal(true);
                }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  background: '#111827',
                  color: '#ffffff',
                  border: 'none',
                  borderRadius: '10px',
                  padding: '6px 14px',
                  fontSize: '12.5px',
                  fontWeight: '700',
                  cursor: 'pointer'
                }}
              >
                <User size={14} />
                <span>Sign In</span>
              </button>
            )}
          </div>
        </header>

        {/* Floating Share Toast Notification */}
        {shareToast && (
          <div style={{
            position: 'fixed',
            top: '20px',
            right: '24px',
            background: '#111827',
            color: '#ffffff',
            padding: '12px 18px',
            borderRadius: '12px',
            boxShadow: '0 12px 30px rgba(0,0,0,0.2)',
            zIndex: 9999,
            fontSize: '13.5px',
            fontWeight: '600',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            animation: 'fadeIn 0.2s ease-out'
          }}>
            <span>{shareToast}</span>
          </div>
        )}

        {/* Scrollable Center Content Area */}
        <div className="main-content-scroll">
          {/* =============================================================== */}
          {/* 1. EVALUATION DASHBOARD & RELIABILITY METRICS (Mode: 'eval')    */}
          {/* =============================================================== */}
          {appMode === 'eval' ? (
            <div className="eval-dashboard-container animate-fade-in">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '10px' }}>
                <div>
                  <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', background: '#eff6ff', color: '#1d4ed8', padding: '4px 12px', borderRadius: '20px', fontSize: '12px', fontWeight: '800', marginBottom: '6px' }}>
                    <BarChart3 size={14} />
                    <span>RAG RELIABILITY & TRUST DASHBOARD</span>
                  </div>
                  <h2 style={{ fontSize: '24px', fontWeight: '800', color: '#111827', margin: 0 }}>
                    RAG Pipeline Evaluation Metrics
                  </h2>
                  <p style={{ fontSize: '13px', color: '#6b7280', margin: '4px 0 0 0' }}>
                    Real-time retrieval accuracy, cosine similarity distributions, confidence tracking, and round-trip latency.
                  </p>
                </div>

                <button 
                  className="btn-download-pdf"
                  onClick={fetchEvalStats}
                  disabled={isLoadingEval}
                  style={{ padding: '8px 14px' }}
                >
                  <RefreshCw size={14} className={isLoadingEval ? 'spin-slow' : ''} />
                  <span>Refresh Metrics</span>
                </button>
              </div>

              {/* 4 Primary KPI Metric Cards */}
              <div className="eval-stats-grid">
                <div className="eval-stat-card">
                  <div className="eval-stat-label">Total Queries Screened</div>
                  <div className="eval-stat-value">{evalStats?.total_queries || 0}</div>
                  <div className="eval-stat-sub">● Live session queries logged</div>
                </div>

                <div className="eval-stat-card">
                  <div className="eval-stat-label">Avg Cosine Similarity</div>
                  <div className="eval-stat-value" style={{ color: '#059669' }}>
                    {evalStats?.avg_top_score ? `${Math.round(evalStats.avg_top_score * 100)}%` : '0%'}
                  </div>
                  <div className="eval-stat-sub">✓ MiniLM-L6 Dense Vectors</div>
                </div>

                <div className="eval-stat-card">
                  <div className="eval-stat-label">High Confidence Rate</div>
                  <div className="eval-stat-value" style={{ color: '#2563eb' }}>
                    {evalStats?.high_confidence_rate || 0}%
                  </div>
                  <div className="eval-stat-sub">● Similarity &ge; 0.70 threshold</div>
                </div>

                <div className="eval-stat-card">
                  <div className="eval-stat-label">Mean Retrieval Latency</div>
                  <div className="eval-stat-value" style={{ color: '#7c3aed' }}>
                    {evalStats?.avg_response_time_ms ? `${Math.round(evalStats.avg_response_time_ms)}ms` : '0ms'}
                  </div>
                  <div className="eval-stat-sub">⚡ ChromaDB + Groq LLM</div>
                </div>
              </div>

              {/* Confidence Level Distribution Segmented Bar */}
              <div className="eval-bar-container">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{ fontSize: '14px', fontWeight: '800', color: '#111827' }}>
                      Confidence Level Distribution
                    </div>
                    <div style={{ fontSize: '12px', color: '#6b7280' }}>
                      Based on Cosine Similarity thresholds against retrieved document chunks
                    </div>
                  </div>
                  <span style={{ fontSize: '12px', fontWeight: '700', color: '#111827' }}>
                    {evalStats?.total_queries || 0} Queries
                  </span>
                </div>

                {/* Visual Segmented Progress Bar */}
                <div className="eval-segmented-bar">
                  <div 
                    className="eval-bar-segment-high" 
                    style={{ width: `${evalStats?.confidence_distribution?.High?.percentage || 0}%` }}
                    title={`High Confidence: ${evalStats?.confidence_distribution?.High?.percentage || 0}%`}
                  />
                  <div 
                    className="eval-bar-segment-mod" 
                    style={{ width: `${evalStats?.confidence_distribution?.Moderate?.percentage || 0}%` }}
                    title={`Moderate Confidence: ${evalStats?.confidence_distribution?.Moderate?.percentage || 0}%`}
                  />
                  <div 
                    className="eval-bar-segment-low" 
                    style={{ width: `${evalStats?.confidence_distribution?.Low?.percentage || 0}%` }}
                    title={`Low / Fallback: ${evalStats?.confidence_distribution?.Low?.percentage || 0}%`}
                  />
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '10px', fontSize: '12px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#22c55e' }} />
                    <span style={{ fontWeight: '700', color: '#166534' }}>
                      🟢 High Confidence: {evalStats?.confidence_distribution?.High?.count || 0} ({evalStats?.confidence_distribution?.High?.percentage || 0}%)
                    </span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#eab308' }} />
                    <span style={{ fontWeight: '700', color: '#854d0e' }}>
                      🟡 Moderate: {evalStats?.confidence_distribution?.Moderate?.count || 0} ({evalStats?.confidence_distribution?.Moderate?.percentage || 0}%)
                    </span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#ef4444' }} />
                    <span style={{ fontWeight: '700', color: '#991b1b' }}>
                      🔴 Low / Fallback: {evalStats?.confidence_distribution?.Low?.count || 0} ({evalStats?.confidence_distribution?.Low?.percentage || 0}%)
                    </span>
                  </div>
                </div>
              </div>

              {/* Chronological Query Latency & Retrieval Score Audit Table */}
              <div className="eval-table-container">
                <div style={{ fontSize: '14px', fontWeight: '800', color: '#111827', marginBottom: '12px' }}>
                  Chronological Query & Retrieval Audit Log
                </div>
                {evalStats?.recent_queries?.length === 0 ? (
                  <div style={{ textAlign: 'center', padding: '30px', color: '#9ca3af', fontSize: '13px' }}>
                    No queries recorded in this session yet. Run a Candidate or Recruiter screening to view real-time metrics!
                  </div>
                ) : (
                  <table className="eval-table">
                    <thead>
                      <tr>
                        <th>Timestamp</th>
                        <th>Query Focus</th>
                        <th>Mode</th>
                        <th>Top Cosine Score</th>
                        <th>Confidence</th>
                        <th>Round-trip Latency</th>
                      </tr>
                    </thead>
                    <tbody>
                      {evalStats?.recent_queries?.map((qItem, qIdx) => (
                        <tr key={qIdx}>
                          <td style={{ color: '#6b7280', fontSize: '11.5px', whiteSpace: 'nowrap' }}>
                            {qItem.timestamp}
                          </td>
                          <td style={{ fontWeight: '600', maxWidth: '300px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {qItem.query}
                          </td>
                          <td>
                            <span style={{ fontSize: '11px', fontWeight: '700', padding: '2px 6px', borderRadius: '4px', background: qItem.mode === 'recruiter' ? '#fef3c7' : '#eff6ff', color: qItem.mode === 'recruiter' ? '#b45309' : '#1d4ed8' }}>
                              {qItem.mode === 'recruiter' ? 'Recruiter' : 'Candidate'}
                            </span>
                          </td>
                          <td style={{ fontWeight: '700', color: qItem.top_score >= 0.7 ? '#059669' : '#d97706' }}>
                            {Math.round(qItem.top_score * 100)}% ({qItem.top_score})
                          </td>
                          <td>
                            <span className={`confidence-badge-pill ${qItem.confidence_label?.toLowerCase().includes('high') ? 'badge-green' : 'badge-yellow'}`} style={{ fontSize: '11px', padding: '2px 8px' }}>
                              {qItem.confidence_label}
                            </span>
                          </td>
                          <td style={{ color: '#4b5563', fontWeight: '600' }}>
                            {qItem.response_time_ms}ms
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>
          ) : appMode === 'recruiter' ? (
            /* =============================================================== */
            /* 2. RECRUITER MODE VIEW (Leaderboard & Multi-Resume Screening)  */
            /* =============================================================== */
            <div className="animate-fade-in" style={{ padding: '10px 4px' }}>
              {leaderboard.length === 0 && !recruiterAnalysis ? (
                /* Recruiter Mode Landing / Empty State */
                <div className="hero-container">
                  <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', background: '#fef3c7', color: '#b45309', padding: '6px 14px', borderRadius: '20px', fontSize: '13px', fontWeight: '700', marginBottom: '14px' }}>
                    <Trophy size={16} />
                    <span>Recruiter Candidate Screening & Leaderboard Mode</span>
                  </div>
                  <h1 className="hero-title">Rank Multiple Candidates for 1 Job</h1>
                  <h2 className="hero-subheading">Side-by-side technical evaluation & candidate leaderboard.</h2>
                  <p className="hero-desc">
                    Upload 1 Target Job Description and up to 10 Candidate Resumes to generate an AI-ranked leaderboard with match scores, skill strengths, gaps, and interview prep questions.
                  </p>

                  <div className="suggestion-cards-grid" style={{ marginTop: '24px' }}>
                    <div className="suggestion-card" onClick={() => document.getElementById('recruiter-jd-upload')?.click()}>
                      <div className="card-icon-lime">
                        <Briefcase size={18} />
                      </div>
                      <div className="card-title">1. Upload Target JD</div>
                      <div className="card-desc">
                        {recruiterJdFile ? `Loaded: ${recruiterJdFile.name}` : "Upload Job Description PDF or paste requirement text."}
                      </div>
                    </div>

                    <div className="suggestion-card" onClick={() => document.getElementById('recruiter-resumes-upload')?.click()}>
                      <div className="card-icon-lime">
                        <Users size={18} />
                      </div>
                      <div className="card-title">2. Upload Candidate Resumes</div>
                      <div className="card-desc">
                        {recruiterResumes.length > 0 ? `${recruiterResumes.length} resume(s) selected` : "Select up to 10 candidate resume PDFs in a batch."}
                      </div>
                    </div>

                    <div 
                      className="suggestion-card" 
                      onClick={handleRecruiterRank}
                      style={{ cursor: (recruiterJdFile || recruiterJdText) && recruiterResumes.length > 0 ? 'pointer' : 'default', opacity: (recruiterJdFile || recruiterJdText) && recruiterResumes.length > 0 ? 1 : 0.6 }}
                    >
                      <div className="card-icon-lime">
                        <Trophy size={18} />
                      </div>
                      <div className="card-title">3. Generate Leaderboard</div>
                      <div className="card-desc">
                        {isRanking ? "Analyzing candidates with Groq..." : "Click to evaluate and rank all candidates from best to weakest fit."}
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                /* Recruiter Leaderboard Active Results */
                <div>
                  <div className="leaderboard-header">
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '10px' }}>
                      <div>
                        <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', background: '#fef3c7', color: '#b45309', padding: '4px 12px', borderRadius: '20px', fontSize: '12px', fontWeight: '800', marginBottom: '8px' }}>
                          <Trophy size={14} />
                          <span>OFFICIAL CANDIDATE RANKING</span>
                        </div>
                        <h2 className="leaderboard-title">Candidate Fit Leaderboard</h2>
                        <p className="leaderboard-subtitle">
                          Evaluated against <strong>{recruiterJdFile?.name || "Target Job Description"}</strong> • {leaderboard.length} Candidates Screened
                        </p>
                      </div>

                      {/* Recruiter PDF Download Button */}
                      <button 
                        className="btn-download-pdf"
                        onClick={() => handleDownloadPdf('recruiter')}
                        disabled={isExportingPdf}
                        title="Export complete leaderboard as PDF"
                      >
                        <Download size={15} />
                        <span>{isExportingPdf ? 'Generating PDF...' : 'Download Leaderboard (PDF)'}</span>
                      </button>
                    </div>

                    {/* 🛡️ Fairness & Bias Audit Notice Panel */}
                    <div className="fairness-notice-box" style={{ marginTop: '16px', textAlign: 'left' }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '6px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', fontWeight: '800', color: '#166534' }}>
                          <ShieldCheck size={17} />
                          <span>Algorithmic Fairness & Bias Audit Notice</span>
                        </div>
                        <span className="fairness-pill">
                          ✓ Programmatically Enforced
                        </span>
                      </div>
                      <p style={{ fontSize: '12px', color: '#14532d', margin: '2px 0 0 0', lineHeight: '1.45' }}>
                        {fairnessAudit?.fairness_notice || (
                          "This ranking is based solely on skills and experience extracted from resumes. Personal signals (age, gender, marital status, nationality) were programmatically excluded from scoring."
                        )}
                      </p>

                      {/* List of candidates with flagged sensitive signals (for disclosure/transparency) */}
                      {fairnessAudit?.flagged_candidates_count > 0 && (
                        <div style={{ marginTop: '4px', display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
                          <span style={{ fontSize: '11px', color: '#92400e', fontWeight: '700' }}>Flagged Demographic Signals:</span>
                          {Object.entries(fairnessAudit.candidate_audits || {}).map(([cName, auditInfo], aIdx) => (
                            auditInfo.contains_sensitive_signals && (
                              <span key={aIdx} className="fairness-flag-pill">
                                ℹ️ {cName}: {auditInfo.flagged_signal_types.join(', ')}
                              </span>
                            )
                          ))}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Leaderboard Candidate Cards */}
                  <div className="leaderboard-grid">
                    {leaderboard.map((cand, cIdx) => {
                      const rankClass = cand.rank === 1 ? 'rank-1' : (cand.rank === 2 ? 'rank-2' : (cand.rank === 3 ? 'rank-3' : 'rank-other'));
                      const rankMedal = cand.rank === 1 ? '🥇 #1' : (cand.rank === 2 ? '🥈 #2' : (cand.rank === 3 ? '🥉 #3' : `#${cand.rank}`));
                      const scoreColor = cand.score >= 85 ? '#10b981' : (cand.score >= 70 ? '#3b82f6' : (cand.score >= 50 ? '#f59e0b' : '#ef4444'));
                      const verdictClass = cand.verdict?.toLowerCase().includes('top') 
                        ? 'verdict-top-pick' 
                        : (cand.verdict?.toLowerCase().includes('short') ? 'verdict-shortlisted' : (cand.verdict?.toLowerCase().includes('consider') ? 'verdict-consider' : 'verdict-mismatch'));

                      return (
                        <div key={cIdx} className="candidate-card animate-fade-in">
                          <div className="candidate-card-top">
                            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                              <span className={`candidate-rank-badge ${rankClass}`}>
                                {rankMedal}
                              </span>
                              <span style={{ fontSize: '16px', fontWeight: '800', color: '#111827' }}>
                                {cand.name}
                              </span>
                            </div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                              <span className={`verdict-tag ${verdictClass}`}>
                                {cand.verdict}
                              </span>
                            </div>
                          </div>

                          {/* Sensitive Signals Transparency Pill on Card */}
                          {cand.contains_sensitive_signals && cand.flagged_signal_types?.length > 0 && (
                            <div style={{ marginBottom: '6px' }}>
                              <span className="fairness-flag-pill" title="Personal indicators detected and excluded from scoring">
                                🛡️ Personal Signals ({cand.flagged_signal_types.join(', ')}) • Excluded from Scoring
                              </span>
                            </div>
                          )}

                          {/* Match Score Progress Bar */}
                          <div className="score-progress-container">
                            <div style={{ fontSize: '13px', fontWeight: '700', color: '#374151', minWidth: '90px' }}>
                              Fit Score: <span style={{ color: scoreColor }}>{cand.score}%</span>
                            </div>
                            <div className="score-bar-bg">
                              <div className="score-bar-fill" style={{ width: `${cand.score}%`, background: scoreColor }} />
                            </div>
                          </div>

                          {/* Why to Select this Candidate */}
                          {cand.why_select && (
                            <div style={{ marginBottom: '10px', background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: '10px', padding: '9px 12px' }}>
                              <div style={{ fontSize: '12px', fontWeight: '800', color: '#166534', display: 'flex', alignItems: 'center', gap: '5px' }}>
                                🎯 Why Select This Candidate:
                              </div>
                              <div style={{ fontSize: '12.5px', color: '#14532d', marginTop: '2px', lineHeight: '1.45' }}>
                                {cand.why_select}
                              </div>
                            </div>
                          )}

                          {/* Why Not to Select / Why Ranked Lower */}
                          {cand.why_not_select && (
                            <div style={{ marginBottom: '10px', background: '#fffbeb', border: '1px solid #fde68a', borderRadius: '10px', padding: '9px 12px' }}>
                              <div style={{ fontSize: '12px', fontWeight: '800', color: '#92400e', display: 'flex', alignItems: 'center', gap: '5px' }}>
                                ⚠️ Deficits & Gaps / Why Ranked Lower:
                              </div>
                              <div style={{ fontSize: '12.5px', color: '#78350f', marginTop: '2px', lineHeight: '1.45' }}>
                                {cand.why_not_select}
                              </div>
                            </div>
                          )}

                          {/* Matching Strengths */}
                          {cand.strengths && cand.strengths.length > 0 && (
                            <div style={{ marginBottom: '8px' }}>
                              <div style={{ fontSize: '12px', fontWeight: '700', color: '#166534', marginBottom: '4px' }}>
                                ✅ Key Strengths:
                              </div>
                              <div className="skills-pill-group">
                                {cand.strengths.map((st, sI) => (
                                  <span key={sI} className="skill-tag-green">
                                    {st}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Skill Gaps */}
                          {cand.gaps && cand.gaps.length > 0 && (
                            <div style={{ marginBottom: '8px' }}>
                              <div style={{ fontSize: '12px', fontWeight: '700', color: '#991b1b', marginBottom: '4px' }}>
                                ❌ Missing Requirements:
                              </div>
                              <div className="skills-pill-group">
                                {cand.gaps.map((gp, gI) => (
                                  <span key={gI} className="skill-tag-red">
                                    {gp}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Actionable Interview Tip */}
                          {cand.recommendation && (
                            <div style={{ marginTop: '10px', background: '#f9fafb', border: '1px solid rgba(0,0,0,0.06)', borderRadius: '8px', padding: '8px 12px', fontSize: '12.5px', color: '#4b5563' }}>
                              <strong style={{ color: '#111827' }}>💡 Actionable Interview Strategy:</strong> {cand.recommendation}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>

                  {/* Detailed Recruiter Analysis Breakdown */}
                  {recruiterAnalysis && (
                    <div style={{ background: '#ffffff', border: '1px solid rgba(0,0,0,0.08)', borderRadius: '16px', padding: '20px', marginTop: '16px', boxShadow: '0 4px 16px rgba(0,0,0,0.03)' }}>
                      <h3 style={{ fontSize: '16px', fontWeight: '800', color: '#111827', marginBottom: '12px' }}>
                        📋 Detailed Hiring Manager Breakdown
                      </h3>
                      <div className="ai-answer-body">
                        {renderMarkdown(recruiterAnalysis)}
                      </div>
                    </div>
                  )}
                  {/* Recruiter Follow-up Inquiries Stream */}
                  {recruiterMessages.length > 0 && (
                    <div style={{ marginTop: '24px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
                      <h3 style={{ fontSize: '15px', fontWeight: '800', color: '#111827', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        💬 Recruiter Q&A & Comparative Deep Dive
                      </h3>
                      {recruiterMessages.map((rMsg, rIdx) => (
                        <React.Fragment key={rIdx}>
                          {rMsg.role === 'user' ? (
                            <div className="chat-message-user animate-fade-in" style={{ color: '#ffffff', fontWeight: '500' }}>
                              {rMsg.content}
                            </div>
                          ) : (
                            <div className="chat-message-ai animate-fade-in">
                              <div className="confidence-banner" style={{ marginBottom: '8px' }}>
                                <span className="confidence-badge-pill badge-green">🟢 Recruiter Assistant</span>
                                <span style={{ fontSize: '11.5px', color: '#9ca3af' }}>{rMsg.timestamp}</span>
                              </div>
                              <div className="ai-answer-body">
                                {renderMarkdown(rMsg.answer)}
                              </div>
                            </div>
                          )}
                        </React.Fragment>
                      ))}
                    </div>
                  )}

                  {isRecruiterAnswering && (
                    <div className="chat-message-ai" style={{ display: 'flex', alignItems: 'center', gap: '10px', flexDirection: 'row', marginTop: '12px' }}>
                      <RefreshCw size={18} className="spin-slow" style={{ color: '#000000' }} />
                      <span style={{ fontSize: '14px', color: '#4b5563' }}>
                        Evaluating candidate batch for your inquiry...
                      </span>
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : (
            /* =============================================================== */
            /* 2. CANDIDATE MODE VIEW (1 Resume vs Multi-JDs & Chat)           */
            /* =============================================================== */
            <>
              {messages.length === 0 ? (
                /* Hero Landing View */
                <div className="hero-container animate-fade-in">
                  <h1 className="hero-title">JD-Fit & Conflict Checker</h1>
                  <h2 className="hero-subheading">Let's make your job search & hiring easier.</h2>
                  <p className="hero-desc">
                    Multi-Document RAG engine for Resume vs. Multi-JD fit assessment, skill gap radar, and automated contradiction detection.
                  </p>

                  {/* Central Floating Input Box */}
                  <div className="floating-input-box">
                    <textarea
                      ref={textareaRef}
                      className="input-textarea"
                      placeholder="Ask anything (e.g. 'Will I get selected for these roles?', 'Compare Role A vs Role B', 'What are my skill gaps?')..."
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault();
                          handleAnalyze();
                        }
                      }}
                    />

                    <div className="input-toolbar">
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <label 
                          htmlFor="sidebar-resume-upload" 
                          className="icon-btn-pill" 
                          style={{ cursor: 'pointer' }}
                          title="Attach resume PDF"
                        >
                          <FileText size={16} />
                        </label>
                        <button 
                          className="icon-btn-pill" 
                          onClick={() => setShowPasteDialog(true)}
                          title="Paste Job Description text"
                        >
                          <Briefcase size={16} />
                        </button>
                        {loadedScope.resume && (
                          <span style={{ fontSize: '11.5px', color: '#059669', fontWeight: '600' }}>
                            ✓ {loadedScope.resume}
                          </span>
                        )}
                      </div>

                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <button 
                          className="btn-send-black"
                          disabled={!query.trim() || isAnalyzing}
                          onClick={() => handleAnalyze()}
                          title="Send query"
                        >
                          <ArrowUp size={18} />
                        </button>
                      </div>
                    </div>
                  </div>

                  {/* Quick Prompts Bar */}
                  <div className="quick-prompts-bar">
                    <button 
                      className="quick-prompt-chip"
                      onClick={() => handleAnalyze("Give an overall selection verdict: Will I get selected for these roles?")}
                    >
                      🎯 Will I get selected?
                    </button>
                    <button 
                      className="quick-prompt-chip"
                      onClick={() => handleAnalyze("What are the key skill gaps, missing tools, or experience differences in my resume?")}
                    >
                      ⚡ What skills am I missing?
                    </button>
                    <button 
                      className="quick-prompt-chip"
                      onClick={() => handleAnalyze("Are there any conflicting requirements between the uploaded Job Descriptions?")}
                    >
                      ⚠️ Check for JD conflicts
                    </button>
                  </div>

                  {/* Suggestion Feature Cards */}
                  <div className="suggestion-cards-grid">
                    <div 
                      className="suggestion-card" 
                      onClick={() => handleAnalyze("Give an overall selection verdict: Will I get selected for these roles?")}
                    >
                      <div className="card-icon-lime">
                        <CheckCircle2 size={18} />
                      </div>
                      <div className="card-title">Overall Selection Verdict</div>
                      <div className="card-desc">Calculate honest match percentage and hiring likelihood across uploaded roles.</div>
                    </div>

                    <div 
                      className="suggestion-card" 
                      onClick={() => handleAnalyze("What are the key skill gaps, missing tools, or experience differences in my resume?")}
                    >
                      <div className="card-icon-lime">
                        <Zap size={18} />
                      </div>
                      <div className="card-title">Skill Gap Radar</div>
                      <div className="card-desc">Identify missing frameworks, experience deficits, and actionable interview tips.</div>
                    </div>

                    <div 
                      className="suggestion-card" 
                      onClick={() => handleAnalyze("Detect and surface any contradictory or conflicting requirements between the uploaded Job Descriptions.")}
                    >
                      <div className="card-icon-lime">
                        <AlertTriangle size={18} />
                      </div>
                      <div className="card-title">Cross-JD Conflict Detection</div>
                      <div className="card-desc">Surface contradictory requirements (e.g. experience years, conflicting tech stacks) across JDs.</div>
                    </div>
                  </div>
                </div>
              ) : (
                /* Conversation & Analysis Stream View */
                <div className="chat-stream-container">
                  {messages.map((msg, idx) => (
                    <React.Fragment key={idx}>
                      {msg.role === 'user' ? (
                        <div className="chat-message-user animate-fade-in">
                          {msg.content}
                        </div>
                      ) : (
                        <div className="chat-message-ai animate-fade-in">
                          {/* Confidence & Model Header */}
                          <div className="confidence-banner">
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                              <span className={`confidence-badge-pill ${msg.confidence_color === 'green' ? 'badge-green' : 'badge-yellow'}`}>
                                {msg.confidence_color === 'green' ? '🟢' : '🟡'} {msg.confidence_label}
                              </span>
                              {msg.top_score > 0 && (
                                <span style={{ fontSize: '12px', color: '#6b7280', fontWeight: '500' }}>
                                  (Similarity: {Math.round(msg.top_score * 100)}%)
                                </span>
                              )}
                            </div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                              <button 
                                className="btn-download-pdf" 
                                style={{ padding: '3px 9px', fontSize: '11.5px' }}
                                onClick={() => handleDownloadPdf('candidate')}
                                disabled={isExportingPdf}
                                title="Download this report as PDF"
                              >
                                <Download size={13} />
                                <span>{isExportingPdf ? 'Exporting...' : 'PDF'}</span>
                              </button>
                              <span style={{ fontSize: '11.5px', color: '#9ca3af' }}>{msg.timestamp}</span>
                            </div>
                          </div>

                          {/* Conflict Detection Banner Callout */}
                          {msg.conflicts && (
                            <div className="conflict-callout-box">
                              <AlertTriangle size={22} style={{ color: '#d97706', flexShrink: 0, marginTop: '2px' }} />
                              <div>
                                <div className="conflict-callout-title">
                                  ⚠️ Conflicts Detected Across Job Descriptions
                                </div>
                                <div className="conflict-callout-content">
                                  {msg.conflicts}
                                </div>
                              </div>
                            </div>
                          )}

                          {/* Main Recruiter Analysis Body */}
                          <div className="ai-answer-body">
                            {renderMarkdown(msg.answer)}
                          </div>

                          {/* Grouped Source Verification Inspector */}
                          {msg.grouped_sources && Object.keys(msg.grouped_sources).length > 0 && (
                            <div className="sources-section-box">
                              <div style={{ fontSize: '13px', fontWeight: '700', color: '#374151', marginBottom: '8px' }}>
                                🔍 Retrieved Grounding Chunks ({Object.keys(msg.grouped_sources).length} Document Sources):
                              </div>
                              <div className="sources-grid">
                                {Object.entries(msg.grouped_sources).map(([docName, chunks], sIdx) => {
                                  const isExpanded = expandedSources[docName] ?? (sIdx === 0);
                                  return (
                                    <div key={docName} className="source-item-card">
                                      <div 
                                        className="sources-toggle-btn" 
                                        onClick={() => toggleSourceGroup(docName)}
                                      >
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                          <FileText size={14} style={{ color: '#4b5563' }} />
                                          <span>{docName}</span>
                                          <span style={{ fontSize: '11px', color: '#9ca3af', fontWeight: '400' }}>
                                            ({chunks.length} chunk{chunks.length > 1 ? 's' : ''})
                                          </span>
                                        </div>
                                        {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                                      </div>

                                      {isExpanded && (
                                        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '6px' }}>
                                          {chunks.map((c, cIdx) => (
                                            <div key={cIdx} className="source-item-quote">
                                              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '3px', fontWeight: '600' }}>
                                                <span>Page {c.page_number} ({c.source_type.toUpperCase()})</span>
                                                <span style={{ color: '#059669' }}>{c.similarity_percentage} match</span>
                                              </div>
                                              <div>"{c.source_text}"</div>
                                            </div>
                                          ))}
                                        </div>
                                      )}
                                    </div>
                                  );
                                })}
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </React.Fragment>
                  ))}

                  {isAnalyzing && (
                    <div className="chat-message-ai" style={{ display: 'flex', alignItems: 'center', gap: '10px', flexDirection: 'row' }}>
                      <RefreshCw size={18} className="spin-slow" style={{ color: '#000000' }} />
                      <span style={{ fontSize: '14px', color: '#4b5563' }}>
                        Retrieving grounding chunks and performing recruiter fit analysis...
                      </span>
                    </div>
                  )}

                  <div ref={chatEndRef} />
                </div>
              )}
            </>
          )}
        </div>

        {/* Docked Floating Input Bar when in chat mode or recruiter results mode */}
        {((appMode === 'candidate' && messages.length > 0) || (appMode === 'recruiter' && (leaderboard.length > 0 || recruiterAnalysis))) && (
          <div style={{ padding: '0 36px 20px 36px', width: '100%', display: 'flex', justifyContent: 'center' }}>
            <div className="floating-input-box" style={{ maxWidth: '840px', width: '100%' }}>
              <textarea
                className="input-textarea"
                placeholder={appMode === 'recruiter' 
                  ? "Ask anything about these candidates (e.g. 'Who has more NLP experience?', 'Draft an interview invitation email for Candidate #1')..." 
                  : "Ask a follow-up question (e.g. 'How should I tailor my resume for Role B?')..."}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    if (appMode === 'recruiter') {
                      handleRecruiterFollowUp();
                    } else {
                      handleAnalyze();
                    }
                  }
                }}
                rows={1}
              />

              <div className="input-toolbar">
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  {appMode === 'recruiter' ? (
                    <span style={{ fontSize: '11.5px', color: '#b45309', fontWeight: '700', background: '#fef3c7', padding: '3px 8px', borderRadius: '6px' }}>
                      🏆 Recruiter Mode • {leaderboard.length} Candidates Loaded
                    </span>
                  ) : (
                    <>
                      {loadedScope.resume && (
                        <span style={{ fontSize: '11.5px', color: '#4b5563', fontWeight: '600' }}>
                          📄 {loadedScope.resume}
                        </span>
                      )}
                      <button 
                        onClick={() => setShowPasteDialog(true)}
                        style={{
                          background: 'none',
                          border: 'none',
                          color: '#059669',
                          fontSize: '11.5px',
                          fontWeight: '600',
                          cursor: 'pointer'
                        }}
                      >
                        + Paste JD
                      </button>
                    </>
                  )}
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <button 
                    className="btn-send-black"
                    disabled={!query.trim() || isAnalyzing || isRecruiterAnswering}
                    onClick={() => {
                      if (appMode === 'recruiter') {
                        handleRecruiterFollowUp();
                      } else {
                        handleAnalyze();
                      }
                    }}
                    title="Send inquiry"
                  >
                    <ArrowUp size={18} />
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* ------------------------------------------------------------------- */}
      {/* DEDICATED DIALOG BOX: COPY-PASTE JOB DESCRIPTION TEXT               */}
      {/* ------------------------------------------------------------------- */}
      {showPasteDialog && (
        <div className="jd-dialog-backdrop" onClick={() => setShowPasteDialog(false)}>
          <div className="jd-dialog-card" onClick={e => e.stopPropagation()}>
            {/* Header */}
            <div className="jd-dialog-header">
              <div className="jd-dialog-title">
                <Edit3 size={20} style={{ color: '#000000' }} />
                <span>Paste Job Description (JD) Text</span>
              </div>
              <button 
                className="icon-btn-pill" 
                onClick={() => setShowPasteDialog(false)}
                title="Close dialog"
              >
                <X size={18} />
              </button>
            </div>

            {/* Body */}
            <div className="jd-dialog-body">
              <div>
                <label style={{ fontSize: '13px', fontWeight: '700', color: '#111827', display: 'block', marginBottom: '6px' }}>
                  Job / Role Title (Optional):
                </label>
                <input 
                  type="text" 
                  placeholder="e.g. Senior Machine Learning Engineer (or Role A)"
                  value={newJdName}
                  onChange={(e) => setNewJdName(e.target.value)}
                  className="jd-dialog-input"
                />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                  <label style={{ fontSize: '13px', fontWeight: '700', color: '#111827' }}>
                    Paste JD Requirements, Responsibilities & Qualifications:
                  </label>
                  <span style={{ fontSize: '11.5px', color: '#6b7280' }}>
                    {newJdText.length} characters ({newJdText.trim() ? newJdText.trim().split(/\s+/).length : 0} words)
                  </span>
                </div>
                <textarea 
                  placeholder="Paste the full job description here... e.g.:

Responsibilities:
- Build and scale production LLM / RAG pipelines.
- 5+ years of experience with Python, PyTorch, and distributed training.
- Strong knowledge of vector databases (ChromaDB, Pinecone).

Qualifications:
- BS/MS in Computer Science or equivalent.
- Experience with FastAPI, Docker, and Kubernetes."
                  value={newJdText}
                  onChange={(e) => setNewJdText(e.target.value)}
                  className="jd-dialog-textarea"
                  rows={8}
                  autoFocus
                />
              </div>

              {/* Information hint */}
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                background: '#f3f4f6',
                borderRadius: '10px',
                padding: '10px 14px',
                fontSize: '12px',
                color: '#4b5563'
              }}>
                <Info size={16} style={{ color: '#2563eb', flexShrink: 0 }} />
                <span>
                  You can paste multiple Job Descriptions. The conflict detection engine will compare all requirements against your resume and surface contradictions.
                </span>
              </div>

              {/* Already added pasted JDs preview */}
              {pastedJds.length > 0 && (
                <div>
                  <div style={{ fontSize: '12px', fontWeight: '700', color: '#374151', marginBottom: '6px' }}>
                    Currently Added Pasted JDs ({pastedJds.length}):
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    {pastedJds.map((jd, idx) => (
                      <div key={idx} style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        background: '#f9fafb',
                        border: '1px solid rgba(0,0,0,0.06)',
                        borderRadius: '8px',
                        padding: '6px 12px',
                        fontSize: '12.5px'
                      }}>
                        <span style={{ fontWeight: '600', maxWidth: '300px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          💼 {jd.name}
                        </span>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <button 
                            type="button"
                            onClick={() => {
                              navigator.clipboard.writeText(jd.text);
                              setShareToast(`📋 Copied "${jd.name}" text to clipboard!`);
                              setTimeout(() => setShareToast(null), 3000);
                            }}
                            style={{ background: 'none', border: 'none', color: '#2563eb', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px', fontSize: '11.5px', fontWeight: '600' }}
                            title="Copy this JD text"
                          >
                            <Copy size={13} />
                            <span>Copy JD</span>
                          </button>
                          <button 
                            type="button"
                            onClick={() => setPastedJds(prev => prev.filter((_, i) => i !== idx))}
                            style={{ background: 'none', border: 'none', color: '#ef4444', cursor: 'pointer' }}
                            title="Delete JD"
                          >
                            <Trash2 size={13} />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Footer */}
            <div className="jd-dialog-footer">
              <button 
                onClick={() => setShowPasteDialog(false)}
                style={{
                  background: '#f3f4f6',
                  border: 'none',
                  borderRadius: '12px',
                  padding: '10px 18px',
                  fontSize: '13px',
                  fontWeight: '600',
                  color: '#374151',
                  cursor: 'pointer'
                }}
              >
                Cancel
              </button>

              <button 
                onClick={handleAddPastedJd}
                disabled={!newJdText.trim()}
                style={{
                  background: '#d4ff00',
                  color: '#000000',
                  border: 'none',
                  borderRadius: '12px',
                  padding: '10px 22px',
                  fontSize: '13.5px',
                  fontWeight: '700',
                  cursor: newJdText.trim() ? 'pointer' : 'not-allowed',
                  opacity: newJdText.trim() ? 1 : 0.4,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px'
                }}
              >
                <Plus size={16} />
                <span>Add Job Description</span>
              </button>
            </div>
          </div>
        </div>
      )}
      {/* ------------------------------------------------------------------- */}
      {/* HISTORY DRAWER: RECENT ANALYSES & SCREENINGS                       */}
      {/* ------------------------------------------------------------------- */}
      {showHistoryDrawer && (
        <div className="history-modal-overlay" onClick={() => setShowHistoryDrawer(false)}>
          <div className="history-modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="history-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Clock size={18} style={{ color: '#111827' }} />
                <h3 style={{ fontSize: '16px', fontWeight: '800', color: '#111827', margin: 0 }}>
                  Analysis & Screening History
                </h3>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                {history.length > 0 && (
                  <button 
                    onClick={handleClearAllHistory}
                    style={{ background: 'none', border: 'none', color: '#ef4444', fontSize: '12px', fontWeight: '600', cursor: 'pointer' }}
                    title="Clear all history records"
                  >
                    Clear All
                  </button>
                )}
                <button 
                  className="file-remove-btn" 
                  onClick={() => setShowHistoryDrawer(false)}
                  title="Close History"
                >
                  <X size={16} />
                </button>
              </div>
            </div>

            <div className="history-list-scroll">
              {history.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '60px 20px', color: '#9ca3af' }}>
                  <Clock size={36} style={{ margin: '0 auto 12px auto', strokeWidth: 1.5, opacity: 0.5 }} />
                  <div style={{ fontSize: '14px', fontWeight: '600', color: '#374151' }}>No History Yet</div>
                  <div style={{ fontSize: '12px', marginTop: '4px' }}>
                    Run a Candidate Fit evaluation or Recruiter Leaderboard to automatically save history records.
                  </div>
                </div>
              ) : (
                history.map((hItem) => (
                  <div 
                    key={hItem.id} 
                    className="history-item-card"
                    onClick={() => handleRestoreHistory(hItem)}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <span className={hItem.type === 'candidate' ? 'history-tag-candidate' : 'history-tag-recruiter'}>
                        {hItem.type === 'candidate' ? <User size={11} /> : <Trophy size={11} />}
                        <span>{hItem.type === 'candidate' ? 'Candidate Fit' : 'Recruiter Leaderboard'}</span>
                      </span>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <span style={{ fontSize: '11px', color: '#9ca3af' }}>{hItem.timestamp}</span>
                        <button 
                          className="file-remove-btn" 
                          onClick={(e) => handleDeleteHistoryItem(e, hItem.id)}
                          title="Delete this record"
                        >
                          <Trash2 size={12} />
                        </button>
                      </div>
                    </div>

                    <div style={{ fontSize: '13.5px', fontWeight: '700', color: '#111827', marginTop: '2px', lineHeight: '1.4' }}>
                      {hItem.title}
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '4px' }}>
                      <span style={{ fontSize: '11.5px', color: '#6b7280' }}>
                        {hItem.type === 'candidate' 
                          ? `${hItem.messages?.length || 0} messages` 
                          : `${hItem.leaderboard?.length || 0} candidates ranked`}
                      </span>
                      <span style={{ fontSize: '11.5px', fontWeight: '700', color: '#2563eb' }}>
                        Load & View →
                      </span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}
      {/* ------------------------------------------------------------------- */}
      {/* AUTHENTICATION MODAL (LOGIN & SIGN UP)                             */}
      {/* ------------------------------------------------------------------- */}
      {showAuthModal && (
        <div className="auth-modal-backdrop" onClick={() => setShowAuthModal(false)}>
          <div className="auth-modal-card" onClick={(e) => e.stopPropagation()}>
            <div style={{ padding: '20px 20px 0 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div>
                <h3 style={{ fontSize: '18px', fontWeight: '800', color: '#111827', margin: 0 }}>
                  {authTab === 'login' ? 'Sign in to EchoAI Fit' : 'Create an Account'}
                </h3>
                <p style={{ fontSize: '12.5px', color: '#6b7280', margin: '4px 0 0 0' }}>
                  Save your analyses, JDs, and candidate leaderboards across logins.
                </p>
              </div>
              <button className="file-remove-btn" onClick={() => setShowAuthModal(false)}>
                <X size={16} />
              </button>
            </div>

            {/* Auth Tabs */}
            <div className="auth-tabs-bar">
              <button 
                className={`auth-tab-btn ${authTab === 'login' ? 'active' : ''}`}
                onClick={() => {
                  setAuthTab('login');
                  setAuthError('');
                }}
              >
                Sign In
              </button>
              <button 
                className={`auth-tab-btn ${authTab === 'register' ? 'active' : ''}`}
                onClick={() => {
                  setAuthTab('register');
                  setAuthError('');
                }}
              >
                New Account
              </button>
            </div>

            {/* Auth Form */}
            <form className="auth-form-body" onSubmit={authTab === 'login' ? handleLogin : handleRegister}>
              {authError && (
                <div style={{ background: '#fef2f2', border: '1px solid #fecaca', color: '#b91c1c', padding: '8px 12px', borderRadius: '8px', fontSize: '12.5px', fontWeight: '500' }}>
                  {authError}
                </div>
              )}

              {authTab === 'register' && (
                <div className="auth-input-group">
                  <label className="auth-input-label">Full Name</label>
                  <input 
                    type="text" 
                    className="auth-input-field" 
                    placeholder="e.g. Madhuri Sharma" 
                    value={authName}
                    onChange={(e) => setAuthName(e.target.value)}
                    required
                  />
                </div>
              )}

              <div className="auth-input-group">
                <label className="auth-input-label">Email Address</label>
                <input 
                  type="email" 
                  className="auth-input-field" 
                  placeholder="e.g. user@example.com" 
                  value={authEmail}
                  onChange={(e) => setAuthEmail(e.target.value)}
                  required
                />
              </div>

              <div className="auth-input-group">
                <label className="auth-input-label">Password</label>
                <input 
                  type="password" 
                  className="auth-input-field" 
                  placeholder="••••••••" 
                  value={authPassword}
                  onChange={(e) => setAuthPassword(e.target.value)}
                  required
                />
              </div>

              {authTab === 'register' && (
                <div className="auth-input-group">
                  <label className="auth-input-label">Primary Role</label>
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <button
                      type="button"
                      onClick={() => setAuthRole('candidate')}
                      style={{
                        flex: 1,
                        padding: '8px',
                        borderRadius: '8px',
                        border: authRole === 'candidate' ? '2px solid #000000' : '1px solid #e5e7eb',
                        background: authRole === 'candidate' ? '#f4f4f5' : '#ffffff',
                        fontSize: '12px',
                        fontWeight: '700',
                        cursor: 'pointer'
                      }}
                    >
                      🎯 Job Seeker
                    </button>
                    <button
                      type="button"
                      onClick={() => setAuthRole('recruiter')}
                      style={{
                        flex: 1,
                        padding: '8px',
                        borderRadius: '8px',
                        border: authRole === 'recruiter' ? '2px solid #000000' : '1px solid #e5e7eb',
                        background: authRole === 'recruiter' ? '#f4f4f5' : '#ffffff',
                        fontSize: '12px',
                        fontWeight: '700',
                        cursor: 'pointer'
                      }}
                    >
                      🏆 Recruiter / HR
                    </button>
                  </div>
                </div>
              )}

              <button type="submit" className="auth-submit-btn">
                {authTab === 'login' ? 'Sign In & Load Workspace' : 'Create Account & Start'}
              </button>

              {/* Quick Demo Login Shortcut */}
              <div style={{ textAlign: 'center', marginTop: '6px' }}>
                <span style={{ fontSize: '11.5px', color: '#9ca3af' }}>Quick Demo Account: </span>
                <button
                  type="button"
                  onClick={() => {
                    setAuthEmail('demo@echoai.com');
                    setAuthPassword('demo123');
                    handleLogin();
                  }}
                  style={{ background: 'none', border: 'none', color: '#2563eb', fontSize: '12px', fontWeight: '700', cursor: 'pointer' }}
                >
                  1-Click Demo Login →
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
