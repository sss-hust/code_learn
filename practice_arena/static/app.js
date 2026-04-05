const state = {
  problems: [],
  filteredProblems: [],
  currentSlug: null,
  currentMode: null,
  dirty: false,
};

const elements = {
  categorySelect: document.getElementById("categorySelect"),
  searchInput: document.getElementById("searchInput"),
  questionList: document.getElementById("questionList"),
  questionCount: document.getElementById("questionCount"),
  runtimeInfo: document.getElementById("runtimeInfo"),
  randomInterviewBtn: document.getElementById("randomInterviewBtn"),
  refreshCatalogBtn: document.getElementById("refreshCatalogBtn"),
  problemTitle: document.getElementById("problemTitle"),
  categoryLabel: document.getElementById("categoryLabel"),
  problemMeta: document.getElementById("problemMeta"),
  modeSelect: document.getElementById("modeSelect"),
  editor: document.getElementById("editor"),
  draftBadge: document.getElementById("draftBadge"),
  problemDescription: document.getElementById("problemDescription"),
  categorySummary: document.getElementById("categorySummary"),
  saveBtn: document.getElementById("saveBtn"),
  resetBtn: document.getElementById("resetBtn"),
  runBtn: document.getElementById("runBtn"),
  testBtn: document.getElementById("testBtn"),
  output: document.getElementById("output"),
  runStatus: document.getElementById("runStatus"),
};

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
      // Ignore JSON parsing errors.
    }
    throw new Error(detail);
  }

  const contentType = response.headers.get("content-type") || "";
  return contentType.includes("application/json") ? response.json() : response.text();
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
    const card = document.createElement("button");
    card.type = "button";
    card.className = `question-card${problem.slug === state.currentSlug ? " active" : ""}`;
    const meta = [problem.category, problem.difficulty, problem.knowledge_points].filter(Boolean).join(" · ");
    card.innerHTML = `
      <h3>${escapeHtml(problem.title)}</h3>
      <p>${escapeHtml(meta || "点击进入练习")}</p>
    `;
    card.addEventListener("click", () => selectProblem(problem.slug, problem.default_mode));
    elements.questionList.appendChild(card);
  }
}

function renderProblem(payload) {
  const { problem, mode, code, draft_exists: draftExists, can_test: canTest } = payload;
  state.currentSlug = problem.slug;
  state.currentMode = mode;
  state.dirty = false;

  elements.categoryLabel.textContent = problem.category;
  elements.problemTitle.textContent = problem.title;
  elements.problemDescription.textContent = problem.description;
  elements.categorySummary.textContent = problem.category_summary;
  elements.editor.value = code;
  elements.draftBadge.textContent = draftExists ? "草稿" : "模板";
  elements.draftBadge.className = draftExists ? "pill" : "pill muted";
  elements.testBtn.disabled = !canTest;

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

  renderQuestionList();
}

function setOutput(text, status = "idle") {
  elements.output.textContent = text;
  elements.runStatus.className = "pill muted";
  elements.runStatus.textContent = status === "running" ? "执行中" : status;
  if (status === "success") {
    elements.runStatus.className = "pill success";
    elements.runStatus.textContent = "通过";
  } else if (status === "failure") {
    elements.runStatus.className = "pill danger";
    elements.runStatus.textContent = "失败";
  } else if (status === "running") {
    elements.runStatus.className = "pill";
  }
}

async function boot() {
  setOutput("正在加载题库...", "running");
  const payload = await request("/api/problems");
  state.problems = payload.problems;
  renderRuntime(payload.runtime);
  populateCategoryOptions();
  applyFilters();

  if (state.problems.length) {
    await selectProblem(state.problems[0].slug, state.problems[0].default_mode, false);
  } else {
    setOutput("没有发现可练习的题目。", "failure");
  }
}

async function selectProblem(slug, mode, shouldSave = true) {
  if (shouldSave) {
    await maybeSaveDraft();
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
  await request("/api/save", {
    method: "POST",
    body: JSON.stringify({
      slug: state.currentSlug,
      mode: state.currentMode,
      code: elements.editor.value,
    }),
  });
  state.dirty = false;
  elements.draftBadge.textContent = "草稿";
  elements.draftBadge.className = "pill";
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
  state.dirty = false;
  elements.draftBadge.textContent = "模板";
  elements.draftBadge.className = "pill muted";
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
    const sections = [
      `$ ${result.command}`,
      `returncode=${result.returncode} duration=${result.duration_ms}ms`,
      `workspace=${result.workspace}`,
      "",
      result.stdout ? `[stdout]\n${result.stdout}` : "[stdout]\n<empty>",
      "",
      result.stderr ? `[stderr]\n${result.stderr}` : "[stderr]\n<empty>",
    ];
    setOutput(sections.join("\n"), result.ok ? "success" : "failure");
    state.dirty = false;
  } catch (error) {
    setOutput(String(error), "failure");
  }
}

async function chooseRandomInterview() {
  const category = elements.categorySelect.value || "all";
  const payload = await request(`/api/random?category=${encodeURIComponent(category)}&mode=interview`);
  await selectProblem(payload.slug, payload.mode);
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
  if (event.key === "Enter") {
    event.preventDefault();
    await runAction("run");
  }
});

boot().catch((error) => {
  setOutput(`初始化失败: ${error}`, "failure");
});
