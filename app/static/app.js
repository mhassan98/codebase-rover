const state = { workspaces: [], models: null, timers: new Map() };

const workspaceSelect = document.querySelector("#workspace");
const scopeNote = document.querySelector("#scope-note");
const modelStatus = document.querySelector("#model-status");

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const text = await response.text();
  let data = null;
  if (text) {
    data = JSON.parse(text);
  }
  if (!response.ok) {
    const detail = data && data.detail ? data.detail : response.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

function selectedWorkspace() {
  return state.workspaces.find((item) => item.slug === workspaceSelect.value) || null;
}

function option(value, label) {
  const node = document.createElement("option");
  node.value = value;
  node.textContent = label;
  return node;
}

function badge(text) {
  const node = document.createElement("span");
  node.className = `badge ${text || ""}`;
  node.textContent = text || "unknown";
  return node;
}

async function loadModels() {
  state.models = await api("/api/models");
  const hostedEnabled = state.models.hosted.enabled;
  modelStatus.textContent = `Ollama chat ${state.models.ollama.chat_model}, embeddings ${state.models.ollama.embed_model}. Hosted models ${hostedEnabled ? "enabled" : "disabled"}.`;
  for (const id of ["ask-model", "review-model"]) {
    const hostedOption = document.querySelector(`#${id} option[value="hosted"]`);
    hostedOption.disabled = !hostedEnabled;
  }
}

async function loadWorkspaces() {
  state.workspaces = await api("/api/workspaces");
  const current = workspaceSelect.value;
  workspaceSelect.replaceChildren();
  for (const workspace of state.workspaces) {
    workspaceSelect.append(option(workspace.slug, `${workspace.name} (${workspace.slug})`));
  }
  if (current && state.workspaces.some((item) => item.slug === current)) {
    workspaceSelect.value = current;
  }
  renderWorkspace();
}

function renderWorkspace() {
  const workspace = selectedWorkspace();
  const environments = document.querySelector("#environments");
  const envSlug = document.querySelector("#env-slug");
  const envRepo = document.querySelector("#env-repo");
  const askRepo = document.querySelector("#ask-repo");
  const reviewRepo = document.querySelector("#review-repo");
  environments.replaceChildren();
  envSlug.replaceChildren();
  envRepo.replaceChildren(option("", "None"));
  askRepo.replaceChildren(option("", "All repositories in this workspace"));
  reviewRepo.replaceChildren(option("", "Diff only, do not read git"));
  if (!workspace) {
    scopeNote.textContent = "No workspace is available.";
    document.querySelector("#repos").replaceChildren();
    return;
  }
  scopeNote.textContent = `Answers and reviews stay in ${workspace.name}. Results from other workspaces are not included.`;
  for (const environment of workspace.environments) {
    const item = document.createElement("li");
    const revision = environment.revision ? `${environment.revision.slice(0, 12)} via ${environment.revision_source}` : "revision not reported";
    item.textContent = `${environment.name}: ${revision}`;
    environments.append(item);
    envSlug.append(option(environment.slug, environment.name));
  }
  for (const repo of workspace.repositories) {
    envRepo.append(option(repo.id, repo.name));
    askRepo.append(option(repo.id, repo.name));
    reviewRepo.append(option(repo.id, repo.name));
  }
  renderRepos(workspace);
}

function renderRepos(workspace) {
  const root = document.querySelector("#repos");
  root.replaceChildren();
  if (!workspace.repositories.length) {
    const empty = document.createElement("p");
    empty.className = "hint";
    empty.textContent = "No repositories connected in this workspace.";
    root.append(empty);
    return;
  }
  for (const repo of workspace.repositories) {
    const card = document.createElement("article");
    card.className = "repo";
    const title = document.createElement("h3");
    title.textContent = repo.name;
    card.append(title, badge(repo.sync_status));
    const meta = document.createElement("p");
    meta.className = "meta";
    const indexed = repo.latest_indexed_commit ? repo.latest_indexed_commit.slice(0, 12) : "not indexed";
    meta.textContent = `${repo.local_path} · indexed ${indexed}`;
    card.append(meta);
    if (repo.last_error) {
      const error = document.createElement("p");
      error.className = "error";
      error.textContent = repo.last_error;
      card.append(error);
    }
    const actions = document.createElement("div");
    actions.className = "row";
    const indexButton = document.createElement("button");
    indexButton.type = "button";
    indexButton.textContent = "Index";
    indexButton.addEventListener("click", () => startIndex(repo.id, false));
    const fullButton = document.createElement("button");
    fullButton.type = "button";
    fullButton.className = "secondary";
    fullButton.textContent = "Full reindex";
    fullButton.addEventListener("click", () => startIndex(repo.id, true));
    const status = document.createElement("p");
    status.className = "meta";
    status.id = `status-${repo.id}`;
    actions.append(indexButton, fullButton);
    card.append(actions, status);
    root.append(card);
  }
}

async function startIndex(repositoryId, force) {
  const workspace = selectedWorkspace();
  const status = document.querySelector(`#status-${repositoryId}`);
  status.textContent = "Starting index…";
  const job = await api(`/api/workspaces/${workspace.slug}/repositories/${repositoryId}/index`, {
    method: "POST",
    body: JSON.stringify({ mode: force ? "full" : "auto", force }),
  });
  pollStatus(workspace.slug, repositoryId, job.id);
}

function pollStatus(slug, repositoryId, jobId) {
  const previous = state.timers.get(repositoryId);
  if (previous) {
    clearInterval(previous);
  }
  const timer = setInterval(async () => {
    try {
      const job = await api(`/api/workspaces/${slug}/index-jobs/${jobId}`);
      const status = document.querySelector(`#status-${repositoryId}`);
      if (status) {
        status.textContent = `${job.status} · mode ${job.resolved_mode || job.requested_mode} · files ${job.files_seen} · chunks ${job.chunks_written}`;
      }
      if (job.status === "succeeded" || job.status === "failed") {
        clearInterval(timer);
        state.timers.delete(repositoryId);
        await loadWorkspaces();
        const live = await api(`/api/workspaces/${slug}/repositories/${repositoryId}/status`);
        const line = document.querySelector(`#status-${repositoryId}`);
        if (line) {
          line.textContent = `${live.sync_status} · head ${short(live.head_commit)} · indexed ${short(live.latest_indexed_commit)} · embeddings ${Math.round(live.embedding_coverage * 100)}%`;
        }
      }
    } catch (error) {
      clearInterval(timer);
      const status = document.querySelector(`#status-${repositoryId}`);
      if (status) {
        status.textContent = error.message;
      }
    }
  }, 1200);
  state.timers.set(repositoryId, timer);
}

function short(value) {
  return value ? value.slice(0, 12) : "none";
}

function renderExcerpts(title, excerpts) {
  const section = document.createElement("section");
  const heading = document.createElement("h3");
  heading.textContent = title;
  section.append(heading);
  if (!excerpts || !excerpts.length) {
    const empty = document.createElement("p");
    empty.className = "hint";
    empty.textContent = "None.";
    section.append(empty);
    return section;
  }
  for (const excerpt of excerpts) {
    const block = document.createElement("article");
    block.className = "excerpt";
    const meta = document.createElement("p");
    meta.className = "meta";
    meta.textContent = `${excerpt.repository_name} ${excerpt.file_path}:${excerpt.start_line}-${excerpt.end_line} @ ${excerpt.commit_sha}`;
    const pre = document.createElement("pre");
    pre.textContent = excerpt.text;
    block.append(meta, pre);
    section.append(block);
  }
  return section;
}

document.querySelector("#repo-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const workspace = selectedWorkspace();
  const ignores = document.querySelector("#repo-ignores").value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  try {
    const repo = await api(`/api/workspaces/${workspace.slug}/repositories`, {
      method: "POST",
      body: JSON.stringify({
        path: document.querySelector("#repo-path").value,
        name: document.querySelector("#repo-name").value || null,
        ignore_globs: ignores,
      }),
    });
    if (document.querySelector("#repo-index-now").checked) {
      await loadWorkspaces();
      await startIndex(repo.id, false);
    } else {
      await loadWorkspaces();
    }
    event.target.reset();
    document.querySelector("#repo-index-now").checked = true;
  } catch (error) {
    alert(error.message);
  }
});

document.querySelector("#ask-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const workspace = selectedWorkspace();
  const target = document.querySelector("#answer");
  target.replaceChildren();
  const pending = document.createElement("p");
  pending.textContent = "Searching this workspace…";
  target.append(pending);
  try {
    const result = await api(`/api/workspaces/${workspace.slug}/questions`, {
      method: "POST",
      body: JSON.stringify({
        question: document.querySelector("#question").value,
        repository_id: document.querySelector("#ask-repo").value || null,
        model_preference: document.querySelector("#ask-model").value,
        confirm_send_context: document.querySelector("#ask-confirm").checked,
      }),
    });
    target.replaceChildren();
    const answer = document.createElement("p");
    answer.textContent = result.answer;
    const meta = document.createElement("p");
    meta.className = "meta";
    meta.textContent = `sufficient: ${result.sufficient} · synthesis: ${result.synthesis}`;
    target.append(answer, meta);
    target.append(renderExcerpts("Citations", result.citations));
    target.append(renderExcerpts("Related excerpts", result.related_excerpts));
  } catch (error) {
    target.replaceChildren();
    const message = document.createElement("p");
    message.className = "error";
    message.textContent = error.message;
    target.append(message);
  }
});

document.querySelector("#review-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const workspace = selectedWorkspace();
  const target = document.querySelector("#review");
  target.replaceChildren();
  const pending = document.createElement("p");
  pending.textContent = "Reviewing the diff…";
  target.append(pending);
  try {
    const result = await api(`/api/workspaces/${workspace.slug}/reviews`, {
      method: "POST",
      body: JSON.stringify({
        title: document.querySelector("#review-title").value || null,
        repository_id: document.querySelector("#review-repo").value || null,
        head_sha: document.querySelector("#review-head").value || null,
        base_sha: document.querySelector("#review-base").value || null,
        diff: document.querySelector("#review-diff").value || null,
        model_preference: document.querySelector("#review-model").value,
        confirm_send_context: document.querySelector("#review-confirm").checked,
      }),
    });
    renderReview(target, result);
  } catch (error) {
    target.replaceChildren();
    const message = document.createElement("p");
    message.className = "error";
    message.textContent = error.message;
    target.append(message);
  }
});

function renderReview(target, result) {
  target.replaceChildren();
  const summary = document.createElement("p");
  summary.textContent = result.summary;
  const meta = document.createElement("p");
  meta.className = "meta";
  meta.textContent = `synthesis: ${result.synthesis} · git verified: ${result.verified_against_git} · head ${result.head_sha || "unverified"}`;
  target.append(summary, meta);
  const files = document.createElement("ul");
  for (const file of result.diff_files || []) {
    const item = document.createElement("li");
    item.textContent = `${file.status} ${file.path} (+${file.added_lines} -${file.deleted_lines})`;
    files.append(item);
  }
  target.append(files);
  const findingsTitle = document.createElement("h3");
  findingsTitle.textContent = "Findings";
  target.append(findingsTitle);
  if (!result.findings.length) {
    const none = document.createElement("p");
    none.className = "hint";
    none.textContent = "No confirmed or likely findings anchored to the diff.";
    target.append(none);
  }
  for (const finding of result.findings) {
    const card = document.createElement("article");
    card.className = "finding";
    card.append(badge(finding.severity), badge(finding.confidence));
    const title = document.createElement("h3");
    title.textContent = finding.title;
    const where = document.createElement("p");
    where.className = "meta";
    where.textContent = `${finding.file_path}:${finding.start_line}-${finding.end_line}`;
    const explanation = document.createElement("p");
    explanation.textContent = finding.explanation;
    card.append(title, where, explanation);
    for (const quote of finding.evidence || []) {
      const pre = document.createElement("pre");
      pre.textContent = quote;
      card.append(pre);
    }
    target.append(card);
  }
  target.append(stringList("Not reported as defects", result.assumptions));
  target.append(stringList("Limitations", result.limitations));
  target.append(renderExcerpts("Context read for this review", result.context_excerpts));
}

function stringList(title, items) {
  const section = document.createElement("section");
  const heading = document.createElement("h3");
  heading.textContent = title;
  section.append(heading);
  if (!items || !items.length) {
    const empty = document.createElement("p");
    empty.className = "hint";
    empty.textContent = "None.";
    section.append(empty);
    return section;
  }
  const list = document.createElement("ul");
  for (const item of items) {
    const li = document.createElement("li");
    li.textContent = item;
    list.append(li);
  }
  section.append(list);
  return section;
}

document.querySelector("#environment-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  await saveEnvironment(false);
});

document.querySelector("#clear-revision").addEventListener("click", () => saveEnvironment(true));

async function saveEnvironment(clearRevision) {
  const workspace = selectedWorkspace();
  try {
    await api(`/api/workspaces/${workspace.slug}/environments/${document.querySelector("#env-slug").value}`, {
      method: "PATCH",
      body: JSON.stringify({
        repository_id: document.querySelector("#env-repo").value || null,
        revision: document.querySelector("#env-revision").value || null,
        clear_revision: clearRevision,
      }),
    });
    await loadWorkspaces();
  } catch (error) {
    alert(error.message);
  }
}

document.querySelector("#refresh").addEventListener("click", () => loadWorkspaces().catch((error) => alert(error.message)));
workspaceSelect.addEventListener("change", renderWorkspace);

document.querySelector("#reset-index").addEventListener("click", async () => {
  if (!confirm("Delete the local code index?")) {
    return;
  }
  try {
    await api("/api/maintenance/reset-index", {
      method: "POST",
      body: JSON.stringify({ confirm: "reset-index" }),
    });
    await loadWorkspaces();
  } catch (error) {
    alert(error.message);
  }
});

loadModels().catch((error) => {
  modelStatus.textContent = error.message;
});
loadWorkspaces().catch((error) => {
  scopeNote.textContent = error.message;
});
