const STATUS_LABELS = {
  not_started: "未开始",
  in_progress: "进行中",
  completed: "已完成",
  review: "待复习",
};

const STATUS_PILL_CLASS = {
  not_started: "pill muted",
  in_progress: "pill warning",
  completed: "pill success",
  review: "pill",
};

const state = {
  problems: [],
  filteredProblems: [],
  progressMap: {},
  currentSlug: null,
  currentMode: null,
  dirty: false,
  canTest: false,
  solutionAvailable: false,
  timer: {
    running: false,
    startedAtMs: null,
    lastSyncAtMs: null,
    sessionSeconds: 0,
    syncing: false,
  },
};

const elements = {
  categorySelect: document.getElementById("categorySelect"),
  searchInput: document.getElementById("searchInput"),
  questionList: document.getElementById("questionList"),
  questionCount: document.getElementById("questionCount"),
  runtimeInfo: document.getElementById("runtimeInfo"),
  dashboardGrid: document.getElementById("dashboardGrid"),
  randomInterviewBtn: document.getElementById("randomInterviewBtn"),
  refreshCatalogBtn: document.getElementById("refreshCatalogBtn"),
  problemTitle: document.getElementById("problemTitle"),
  categoryLabel: document.getElementById("categoryLabel"),
  problemMeta: document.getElementById("problemMeta"),
  modeSelect: document.getElementById("modeSelect"),
  modeLabel: document.getElementById("modeLabel"),
  editor: document.getElementById("editor"),
  draftBadge: document.getElementById("draftBadge"),
  problemDescription: document.getElementById("problemDescription"),
  categorySummary: document.getElementById("categorySummary"),
  saveBtn: document.getElementById("saveBtn"),
  resetBtn: document.getElementById("resetBtn"),
  runBtn: document.getElementById("runBtn"),
  testBtn: document.getElementById("testBtn"),
  evaluateBtn: document.getElementById("evaluateBtn"),
  output: document.getElementById("output"),
  runStatus: document.getElementById("runStatus"),
  timerDisplay: document.getElementById("timerDisplay"),
  totalTimeLabel: document.getElementById("totalTimeLabel"),
  lastScoreLabel: document.getElementById("lastScoreLabel"),
  bestScoreLabel: document.getElementById("bestScoreLabel"),
  timerToggleBtn: document.getElementById("timerToggleBtn"),
  timerStopBtn: document.getElementById("timerStopBtn"),
  statusSelect: document.getElementById("statusSelect"),
  statusBadge: document.getElementById("statusBadge"),
  attemptCountLabel: document.getElementById("attemptCountLabel"),
  lastRunLabel: document.getElementById("lastRunLabel"),
  solutionViewedLabel: document.getElementById("solutionViewedLabel"),
  historyList: document.getElementById("historyList"),
  showSolutionBtn: document.getElementById("showSolutionBtn"),
  compareBtn: document.getElementById("compareBtn"),
  evaluationSummary: document.getElementById("evaluationSummary"),
  reviewOutput: document.getElementById("reviewOutput"),
  reviewBadge: document.getElementById("reviewBadge"),
  markCompletedBtn: document.getElementById("markCompletedBtn"),
  markReviewBtn: document.getElementById("markReviewBtn"),
};

let timerIntervalId = null;

async function request(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      if (payload.detail) {
        detail = payload.detail;
      }
    } catch (_error) {
      // Ignore JSON parsing failures.
    }
    throw new Error(detail);
  }

  const contentType = response.headers.get("content-type") || "";
  return contentType.includes("application/json") ? response.json() : response.text();
}

function defaultProgress() {
  return {
    status: "not_started",
    total_seconds: 0,
    attempts: 0,
    best_score: 0,
    last_score: null,
    last_mode: null,
    last_opened_at: null,
    last_run_at: null,
    last_run_ok: null,
    last_action: null,
    last_duration_ms: null,
    last_returncode: null,
    solution_revealed: false,
    solution_view_count: 0,
    last_solution_view_at: null,
    updated_at: null,
    history: [],
  };
}

function progressFor(slug) {
  return state.progressMap[slug] || defaultProgress();
}

function statusLabel(status) {
  return STATUS_LABELS[status] || status || "未开始";
}

function statusPillClass(status) {
  return STATUS_PILL_CLASS[status] || "pill muted";
}

function formatDuration(totalSeconds) {
  const safeSeconds = Math.max(0, Number(totalSeconds || 0));
  const hours = Math.floor(safeSeconds / 3600);
  const minutes = Math.floor((safeSeconds % 3600) / 60);
  const seconds = safeSeconds % 60;
  return [hours, minutes, seconds].map((value) => String(value).padStart(2, "0")).join(":");
}

function formatTimestamp(value) {
  if (!value) {
    return "--";
  }
  return new Date(value).toLocaleString("zh-CN", {
    hour12: false,
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function renderRuntime(runtime) {
  elements.runtimeInfo.innerHTML = "";
  const entries = [
    ["Python", runtime.python],
    ["pytest", runtime.pytest ? "已安装" : "缺失"],
    ["nvcc", runtime.nvcc ? "可用" : "缺失"],
    ["Repo", runtime.repo_root],
    ["Draft", runtime.draft_root],
  ];

  for (const [label, value] of entries) {
    const row = document.createElement("div");
    row.innerHTML = `<dt>${escapeHtml(label)}</dt><dd>${escapeHtml(String(value))}</dd>`;
    elements.runtimeInfo.appendChild(row);
  }
}

function renderDashboard() {
  const values = Object.values(state.progressMap);
  const summary = {
    completed: values.filter((item) => item.status === "completed").length,
    review: values.filter((item) => item.status === "review").length,
    inProgress: values.filter((item) => item.status === "in_progress").length,
    attempted: values.filter((item) => item.attempts > 0).length,
    totalSeconds: values.reduce((sum, item) => sum + Number(item.total_seconds || 0), 0),
    bestScore: values.reduce((maxValue, item) => Math.max(maxValue, Number(item.best_score || 0)), 0),
  };

  const cards = [
    ["已完成", summary.completed],
    ["待复习", summary.review],
    ["进行中", summary.inProgress],
    ["有提交", summary.attempted],
    ["累计时长", formatDuration(summary.totalSeconds)],
    ["最高分", summary.bestScore ? `${summary.bestScore}` : "--"],
  ];

  elements.dashboardGrid.innerHTML = cards
    .map(
      ([label, value]) => `
        <div class="dashboard-card">
          <span>${escapeHtml(String(label))}</span>
          <strong>${escapeHtml(String(value))}</strong>
        </div>
      `,
    )
    .join("");
}

function populateCategoryOptions() {
  const categories = ["all", ...new Set(state.problems.map((problem) => problem.category))];
  elements.categorySelect.innerHTML = categories
    .map((category) => {
      const label = category === "all" ? "全部专题" : category;
      return `<option value="${escapeHtml(category)}">${escapeHtml(label)}</option>`;
    })
    .join("");
}

function applyFilters() {
  const category = elements.categorySelect.value || "all";
  const keyword = (elements.searchInput.value || "").trim().toLowerCase();
  state.filteredProblems = state.problems.filter((problem) => {
    const categoryMatch = category === "all" || problem.category === category;
    const keywordMatch =
      !keyword ||
      problem.title.toLowerCase().includes(keyword) ||
      (problem.knowledge_points || "").toLowerCase().includes(keyword);
    return categoryMatch && keywordMatch;
  });
  renderQuestionList();
}

function renderQuestionList() {
  elements.questionCount.textContent = String(state.filteredProblems.length);
  elements.questionList.innerHTML = "";

  if (!state.filteredProblems.length) {
    elements.questionList.innerHTML = "<p class='subcopy'>没有匹配题目。</p>";
    return;
  }

  for (const problem of state.filteredProblems) {
    const progress = progressFor(problem.slug);
    const card = document.createElement("button");
    card.type = "button";
    card.className = `question-card${problem.slug === state.currentSlug ? " active" : ""}`;
    card.innerHTML = `
      <div class="status-inline">
        <h3>${escapeHtml(problem.title)}</h3>
        <span class="${escapeHtml(statusPillClass(progress.status))}">${escapeHtml(statusLabel(progress.status))}</span>
      </div>
      <p>${escapeHtml([problem.category, problem.difficulty, problem.knowledge_points].filter(Boolean).join(" · ") || "点击进入练习")}</p>
      <p>${escapeHtml(`尝试 ${progress.attempts} 次 · 累计 ${formatDuration(progress.total_seconds)}`)}</p>
    `;
    card.addEventListener("click", () => selectProblem(problem.slug, problem.default_mode));
    elements.questionList.appendChild(card);
  }
}

function renderProblem(payload) {
  const { problem, mode, code, draft_exists: draftExists, can_test: canTest, solution_available: solutionAvailable } = payload;
  state.currentSlug = problem.slug;
  state.currentMode = mode;
  state.dirty = false;
  state.canTest = canTest;
  state.solutionAvailable = solutionAvailable;
  if (payload.progress) {
    state.progressMap[problem.slug] = payload.progress;
  }

  elements.categoryLabel.textContent = problem.category;
  elements.problemTitle.textContent = problem.title;
  elements.problemDescription.textContent = problem.description;
  elements.categorySummary.textContent = problem.category_summary;
  elements.editor.value = code;
  elements.draftBadge.textContent = draftExists ? "草稿" : "模板";
  elements.draftBadge.className = draftExists ? "pill" : "pill muted";
  elements.testBtn.disabled = !canTest;
  elements.showSolutionBtn.disabled = !solutionAvailable;
  elements.compareBtn.disabled = !solutionAvailable;
  elements.modeLabel.textContent = mode;

  const metaItems = [
    problem.language,
    `默认模式: ${problem.default_mode}`,
    problem.knowledge_points ? `知识点: ${problem.knowledge_points}` : null,
    problem.difficulty ? `难度: ${problem.difficulty}` : null,
    problem.test_file ? `测试: ${problem.test_file}` : "测试: 自行运行",
  ].filter(Boolean);
  elements.problemMeta.innerHTML = metaItems.map((item) => `<span class="pill muted">${escapeHtml(item)}</span>`).join("");

  elements.modeSelect.innerHTML = problem.available_modes
    .map((item) => `<option value="${escapeHtml(item)}">${escapeHtml(item)}</option>`)
    .join("");
  elements.modeSelect.value = mode;

  resetReviewPane();
  renderProgress(payload.progress || progressFor(problem.slug));
  renderQuestionList();
}

function renderProgress(progress) {
  const safeProgress = progress || defaultProgress();
  elements.statusSelect.value = safeProgress.status || "not_started";
  elements.statusBadge.textContent = statusLabel(safeProgress.status);
  elements.statusBadge.className = statusPillClass(safeProgress.status);
  elements.attemptCountLabel.textContent = String(safeProgress.attempts || 0);
  elements.lastRunLabel.textContent = safeProgress.last_run_at
    ? `${formatTimestamp(safeProgress.last_run_at)} / ${safeProgress.last_action || "run"}`
    : "--";
  elements.solutionViewedLabel.textContent = String(safeProgress.solution_view_count || 0);
  elements.lastScoreLabel.textContent = safeProgress.last_score == null ? "--" : String(safeProgress.last_score);
  elements.bestScoreLabel.textContent = safeProgress.best_score ? String(safeProgress.best_score) : "--";
  updateTimerDisplay();

  const history = (safeProgress.history || []).slice(-6).reverse();
  elements.historyList.innerHTML = history.length
    ? history
        .map(
          (item) => `
            <div class="history-item">
              <strong>${escapeHtml(item.action)} · ${item.ok ? "OK" : "FAIL"}${item.score != null ? ` · score ${item.score}` : ""}</strong>
              <span>${escapeHtml(formatTimestamp(item.at))} · ${escapeHtml(String(item.duration_ms))} ms · rc=${escapeHtml(String(item.returncode))}</span>
            </div>
          `,
        )
        .join("")
    : "<p class='subcopy'>还没有提交记录。</p>";
}

function resetReviewPane() {
  elements.reviewBadge.textContent = "隐藏";
  elements.reviewBadge.className = "pill muted";
  elements.evaluationSummary.textContent = "评分是启发式的，优先用于提醒 correctness、可运行性和占位符清理情况。";
  elements.reviewOutput.textContent = "等待查看参考答案或评分结果...";
}

function setReviewPanel(text, summary, badgeText, badgeClass = "pill") {
  elements.reviewOutput.textContent = text;
  elements.evaluationSummary.textContent = summary;
  elements.reviewBadge.textContent = badgeText;
  elements.reviewBadge.className = badgeClass;
}

function renderRunResult(result, statusOverride = null) {
  const sections = [
    `$ ${result.command}`,
    `returncode=${result.returncode} duration=${result.duration_ms}ms`,
    `workspace=${result.workspace}`,
    "",
    result.stdout ? `[stdout]\n${result.stdout}` : "[stdout]\n<empty>",
    "",
    result.stderr ? `[stderr]\n${result.stderr}` : "[stderr]\n<empty>",
  ];
  setOutput(sections.join("\n"), statusOverride || (result.ok ? "success" : "failure"));
}

function setOutput(text, status = "idle") {
  elements.output.textContent = text;
  elements.runStatus.className = "pill muted";
  elements.runStatus.textContent = status === "running" ? "执行中" : status === "idle" ? "空闲" : status;
  if (status === "success") {
    elements.runStatus.className = "pill success";
    elements.runStatus.textContent = "通过";
  } else if (status === "failure") {
    elements.runStatus.className = "pill danger";
    elements.runStatus.textContent = "失败";
  } else if (status === "running") {
    elements.runStatus.className = "pill warning";
  } else if (status === "warning") {
    elements.runStatus.className = "pill warning";
    elements.runStatus.textContent = "启发式";
  }
}

async function boot() {
  setOutput("正在加载题库...", "running");
  const payload = await request("/api/problems");
  state.problems = payload.problems;
  state.progressMap = payload.progress || {};
  renderRuntime(payload.runtime);
  renderDashboard();
  populateCategoryOptions();
  applyFilters();

  if (state.problems.length) {
    await selectProblem(state.problems[0].slug, state.problems[0].default_mode, false);
  } else {
    setOutput("没有发现可练习的题目。", "failure");
  }
}

async function selectProblem(slug, mode, shouldSave = true) {
  const switchingProblem = state.currentSlug && state.currentSlug !== slug;
  if (shouldSave) {
    await maybeSaveDraft();
  }
  if (switchingProblem) {
    await flushTimer(true);
    resetTimerState();
  }

  const payload = await request(`/api/problem/${slug}?mode=${encodeURIComponent(mode)}`);
  renderProblem(payload);
  setOutput(`已加载 ${slug} (${mode})。`, "idle");
}

async function maybeSaveDraft() {
  if (!state.currentSlug || !state.dirty) {
    return;
  }
  await saveDraft(false);
}

async function saveDraft(notify = true) {
  if (!state.currentSlug) {
    return;
  }
  const payload = await request("/api/save", {
    method: "POST",
    body: JSON.stringify({
      slug: state.currentSlug,
      mode: state.currentMode,
      code: elements.editor.value,
    }),
  });
  state.progressMap[state.currentSlug] = payload.progress;
  state.dirty = false;
  elements.draftBadge.textContent = "草稿";
  elements.draftBadge.className = "pill";
  renderProgress(payload.progress);
  renderDashboard();
  renderQuestionList();
  if (notify) {
    setOutput("草稿已保存。", "idle");
  }
}

async function resetDraft() {
  if (!state.currentSlug) {
    return;
  }
  if (!window.confirm("恢复模板会删除当前模式下的草稿，是否继续？")) {
    return;
  }
  const payload = await request("/api/reset", {
    method: "POST",
    body: JSON.stringify({
      slug: state.currentSlug,
      mode: state.currentMode,
      code: "",
    }),
  });
  elements.editor.value = payload.code;
  state.progressMap[state.currentSlug] = payload.progress;
  state.dirty = false;
  elements.draftBadge.textContent = "模板";
  elements.draftBadge.className = "pill muted";
  renderProgress(payload.progress);
  renderDashboard();
  renderQuestionList();
  setOutput("已恢复为模板代码。", "idle");
}

async function runAction(action) {
  if (!state.currentSlug) {
    return;
  }
  setOutput("执行中，请稍候...", "running");
  try {
    const result = await request("/api/run", {
      method: "POST",
      body: JSON.stringify({
        slug: state.currentSlug,
        mode: state.currentMode,
        code: elements.editor.value,
        action,
      }),
    });
    state.progressMap[state.currentSlug] = result.progress;
    renderProgress(result.progress);
    renderDashboard();
    renderQuestionList();
    renderRunResult(result);
    state.dirty = false;
  } catch (error) {
    setOutput(String(error), "failure");
  }
}

async function evaluateCurrent() {
  if (!state.currentSlug) {
    return;
  }
  setReviewPanel("评分中...", "正在执行当前代码并生成启发式评分。", "评分中", "pill warning");
  try {
    const result = await request("/api/evaluate", {
      method: "POST",
      body: JSON.stringify({
        slug: state.currentSlug,
        mode: state.currentMode,
        code: elements.editor.value,
      }),
    });
    state.progressMap[state.currentSlug] = result.progress;
    renderProgress(result.progress);
    renderDashboard();
    renderQuestionList();
    renderRunResult(result.run_result, "warning");

    const lines = [
      `总分: ${result.score} / 100`,
      `等级: ${result.grade}`,
      `评估方式: ${result.action_used}`,
      result.similarity == null ? "参考答案相似度: --" : `参考答案相似度: ${result.similarity}`,
      "",
      "评分拆解:",
      ...result.breakdown.map(
        (item) => `- ${item.label}: ${item.score}/${item.max_score} | ${item.detail}`,
      ),
      "",
      result.placeholder_hits.length
        ? `检测到占位符: ${result.placeholder_hits.join(", ")}`
        : "未检测到明显占位符。",
    ];

    setReviewPanel(
      lines.join("\n"),
      result.summary,
      `评分 ${result.score}`,
      result.score >= 75 ? "pill success" : result.score >= 60 ? "pill warning" : "pill danger",
    );
  } catch (error) {
    setReviewPanel(String(error), "评分失败。", "失败", "pill danger");
  }
}

async function showSolution() {
  if (!state.currentSlug || !state.solutionAvailable) {
    return;
  }
  try {
    const payload = await request(`/api/solution/${state.currentSlug}?mode=${encodeURIComponent(state.currentMode)}`);
    state.progressMap[state.currentSlug] = payload.progress;
    renderProgress(payload.progress);
    renderDashboard();
    renderQuestionList();
    setReviewPanel(
      payload.solution_code,
      `参考答案文件: ${payload.solution_file}。查看答案后仍可继续练，但盲写加分会失效。`,
      "答案已展开",
      "pill warning",
    );
  } catch (error) {
    setReviewPanel(String(error), "当前题目没有参考答案。", "失败", "pill danger");
  }
}

async function compareSolution() {
  if (!state.currentSlug || !state.solutionAvailable) {
    return;
  }
  try {
    const payload = await request("/api/compare", {
      method: "POST",
      body: JSON.stringify({
        slug: state.currentSlug,
        mode: state.currentMode,
        code: elements.editor.value,
      }),
    });
    state.progressMap[state.currentSlug] = payload.progress;
    renderProgress(payload.progress);
    renderDashboard();
    renderQuestionList();
    const diffText = payload.diff || "当前代码与参考答案完全一致。";
    setReviewPanel(
      diffText,
      `参考答案: ${payload.solution_file} · 相似度 ${payload.similarity}`,
      "Diff",
      "pill",
    );
  } catch (error) {
    setReviewPanel(String(error), "无法对比参考答案。", "失败", "pill danger");
  }
}

async function chooseRandomInterview() {
  const category = elements.categorySelect.value || "all";
  const payload = await request(`/api/random?category=${encodeURIComponent(category)}&mode=interview`);
  await selectProblem(payload.slug, payload.mode);
}

function getLiveSessionSeconds() {
  if (!state.timer.running || !state.timer.startedAtMs) {
    return state.timer.sessionSeconds;
  }
  return state.timer.sessionSeconds + Math.floor((Date.now() - state.timer.startedAtMs) / 1000);
}

function getUnsyncedSeconds() {
  if (!state.timer.running || !state.timer.lastSyncAtMs) {
    return 0;
  }
  return Math.floor((Date.now() - state.timer.lastSyncAtMs) / 1000);
}

function ensureTimerLoop() {
  if (timerIntervalId != null) {
    return;
  }
  timerIntervalId = window.setInterval(() => {
    updateTimerDisplay();
    if (!state.timer.running) {
      return;
    }
    const unsynced = getUnsyncedSeconds();
    if (unsynced >= 30 && !state.timer.syncing) {
      const delta = unsynced;
      state.timer.syncing = true;
      syncProgress({ seconds_delta: delta })
        .then(() => {
          if (state.timer.lastSyncAtMs != null) {
            state.timer.lastSyncAtMs += delta * 1000;
          }
          updateTimerDisplay();
        })
        .catch((error) => {
          setOutput(`计时同步失败: ${error}`, "failure");
        })
        .finally(() => {
          state.timer.syncing = false;
        });
    }
  }, 1000);
}

function resetTimerState() {
  state.timer.running = false;
  state.timer.startedAtMs = null;
  state.timer.lastSyncAtMs = null;
  state.timer.sessionSeconds = 0;
  state.timer.syncing = false;
  elements.timerToggleBtn.textContent = "开始计时";
  updateTimerDisplay();
}

function updateTimerDisplay() {
  const progress = state.currentSlug ? progressFor(state.currentSlug) : defaultProgress();
  const sessionSeconds = getLiveSessionSeconds();
  const totalSeconds = Number(progress.total_seconds || 0) + getUnsyncedSeconds();
  elements.timerDisplay.textContent = formatDuration(sessionSeconds);
  elements.totalTimeLabel.textContent = formatDuration(totalSeconds);
}

async function toggleTimer() {
  if (!state.currentSlug) {
    return;
  }
  if (!state.timer.running) {
    state.timer.running = true;
    state.timer.startedAtMs = Date.now();
    state.timer.lastSyncAtMs = Date.now();
    elements.timerToggleBtn.textContent = "暂停计时";
    ensureTimerLoop();
    if (elements.statusSelect.value === "not_started") {
      await setStatus("in_progress");
    } else {
      updateTimerDisplay();
    }
    return;
  }
  await flushTimer(true);
}

async function flushTimer(resetAfterFlush) {
  if (!state.timer.running || !state.currentSlug) {
    if (resetAfterFlush) {
      elements.timerToggleBtn.textContent = "开始计时";
    }
    return;
  }
  const fullSessionSeconds = getLiveSessionSeconds();
  const unsynced = getUnsyncedSeconds();
  if (unsynced > 0) {
    await syncProgress({ seconds_delta: unsynced });
  }
  state.timer.sessionSeconds = fullSessionSeconds;
  state.timer.running = false;
  state.timer.startedAtMs = null;
  state.timer.lastSyncAtMs = null;
  elements.timerToggleBtn.textContent = "开始计时";
  updateTimerDisplay();
  if (resetAfterFlush) {
    state.timer.sessionSeconds = 0;
    updateTimerDisplay();
  }
}

async function syncProgress(extraPayload) {
  if (!state.currentSlug) {
    return;
  }
  const payload = await request("/api/progress", {
    method: "POST",
    body: JSON.stringify({
      slug: state.currentSlug,
      mode: state.currentMode,
      ...extraPayload,
    }),
  });
  state.progressMap[state.currentSlug] = payload.progress;
  renderProgress(payload.progress);
  renderDashboard();
  renderQuestionList();
}

async function setStatus(status) {
  if (!state.currentSlug) {
    return;
  }
  await syncProgress({ status });
}

function sendBeaconProgress() {
  if (!state.currentSlug || !state.timer.running) {
    return;
  }
  const unsynced = getUnsyncedSeconds();
  if (unsynced <= 0 || !navigator.sendBeacon) {
    return;
  }
  const payload = {
    slug: state.currentSlug,
    mode: state.currentMode,
    seconds_delta: unsynced,
  };
  navigator.sendBeacon("/api/progress", new Blob([JSON.stringify(payload)], { type: "application/json" }));
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

elements.categorySelect.addEventListener("change", applyFilters);
elements.searchInput.addEventListener("input", applyFilters);
elements.refreshCatalogBtn.addEventListener("click", boot);
elements.randomInterviewBtn.addEventListener("click", chooseRandomInterview);
elements.saveBtn.addEventListener("click", () => saveDraft(true));
elements.resetBtn.addEventListener("click", resetDraft);
elements.runBtn.addEventListener("click", () => runAction("run"));
elements.testBtn.addEventListener("click", () => runAction("test"));
elements.evaluateBtn.addEventListener("click", evaluateCurrent);
elements.showSolutionBtn.addEventListener("click", showSolution);
elements.compareBtn.addEventListener("click", compareSolution);
elements.timerToggleBtn.addEventListener("click", toggleTimer);
elements.timerStopBtn.addEventListener("click", () => flushTimer(false));
elements.statusSelect.addEventListener("change", (event) => {
  void setStatus(event.target.value);
});
elements.markCompletedBtn.addEventListener("click", () => setStatus("completed"));
elements.markReviewBtn.addEventListener("click", () => setStatus("review"));
elements.modeSelect.addEventListener("change", async (event) => {
  if (!state.currentSlug) {
    return;
  }
  await selectProblem(state.currentSlug, event.target.value);
});
elements.editor.addEventListener("input", () => {
  state.dirty = true;
});

window.addEventListener("beforeunload", (event) => {
  sendBeaconProgress();
  if (!state.dirty) {
    return;
  }
  event.preventDefault();
  event.returnValue = "";
});

window.addEventListener("keydown", async (event) => {
  const isMac = navigator.platform.toUpperCase().includes("MAC");
  const primaryKey = isMac ? event.metaKey : event.ctrlKey;
  if (!primaryKey) {
    return;
  }
  if (event.key.toLowerCase() === "s") {
    event.preventDefault();
    await saveDraft(true);
  }
  if (event.key === "Enter" && event.shiftKey) {
    event.preventDefault();
    await evaluateCurrent();
  } else if (event.key === "Enter") {
    event.preventDefault();
    await runAction("run");
  }
});

ensureTimerLoop();
boot().catch((error) => {
  setOutput(`初始化失败: ${error}`, "failure");
});
