(() => {
  const timeline = document.querySelector("[data-events-url]");
  if (!timeline || timeline.dataset.terminal === "true") {
    return;
  }

  const STATUS_LABELS = Object.freeze({
    pending: "待处理",
    running: "运行中",
    waiting_approval: "等待审批",
    completed: "已完成",
    blocked: "已阻止",
    cancelled: "已取消",
    interrupted: "已中断",
  });
  const terminal = new Set(["completed", "blocked", "cancelled", "interrupted"]);
  const list = timeline.querySelector("[data-event-list]");
  const status = timeline.querySelector("[data-task-status]");
  const statusLabel = (value) => STATUS_LABELS[value] || value;

  const detailSpan = (className, text) => {
    const span = document.createElement("span");
    span.className = className;
    span.textContent = text;
    return span;
  };

  const renderEvent = (event) => {
    const item = document.createElement("li");
    item.dataset.eventStatus = event.task_status;
    const header = document.createElement("div");
    header.className = "timeline-event-header";
    const badge = detailSpan("badge", statusLabel(event.task_status));
    badge.dataset.status = event.task_status;
    header.appendChild(badge);
    if (event.action_summary) {
      header.appendChild(detailSpan("event-action", event.action_summary));
    }
    item.appendChild(header);

    const detail = document.createElement("div");
    detail.className = "timeline-event-detail";
    if (event.action_projection) detail.appendChild(detailSpan("event-action mono", event.action_projection));
    if (event.affected_project) detail.appendChild(detailSpan("event-governance mono", `项目：${event.affected_project}`));
    if (event.policy_verdict) detail.appendChild(detailSpan("event-governance mono", `策略：${event.policy_verdict}`));
    if (event.policy_rule_id) detail.appendChild(detailSpan("event-governance mono", `规则：${event.policy_rule_id}`));
    if (event.policy_reason) detail.appendChild(detailSpan("event-governance", event.policy_reason));
    if (event.approval_granted === true) detail.appendChild(detailSpan("event-governance", "审批：已同意"));
    if (event.approval_granted === false) detail.appendChild(detailSpan("event-governance", "审批：已拒绝"));
    if (event.feedback_excerpt) detail.appendChild(detailSpan("event-feedback", event.feedback_excerpt));
    if (event.feedback_node_id) detail.appendChild(detailSpan("event-feedback mono", event.feedback_node_id));
    if (event.stop_reason) detail.appendChild(detailSpan("event-stop mono", `停止：${event.stop_reason}`));
    if (detail.childElementCount) item.appendChild(detail);
    return item;
  };

  const poll = async () => {
    const response = await fetch(timeline.dataset.eventsUrl, { headers: { Accept: "application/json" } });
    if (!response.ok) {
      return;
    }
    const events = await response.json();
    const latest = events.at(-1);
    if (!latest) {
      return;
    }
    if (timeline.dataset.currentStatus !== latest.task_status) {
      window.location.reload();
      return;
    }
    if (status) {
      status.textContent = statusLabel(latest.task_status);
      status.dataset.status = latest.task_status;
    }
    if (list) {
      list.replaceChildren(...events.map(renderEvent));
    }
    if (terminal.has(latest.task_status)) {
      clearInterval(interval);
    }
  };

  const interval = window.setInterval(poll, 2000);
  poll();
})();
