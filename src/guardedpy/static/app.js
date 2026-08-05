(() => {
  const timeline = document.querySelector("[data-events-url]");
  if (!timeline || timeline.dataset.terminal === "true") {
    return;
  }

  const terminal = new Set(["completed", "blocked", "cancelled", "interrupted"]);
  const list = timeline.querySelector("[data-event-list]");
  const status = timeline.querySelector("[data-task-status]");

  const renderEvent = (event) => {
    const item = document.createElement("li");
    const approval = event.approval_granted === true
      ? "approval: granted"
      : event.approval_granted === false
        ? "approval: rejected"
        : null;
    const parts = [
      event.task_status,
      event.action_summary,
      event.action_projection,
      event.affected_project ? `Project: ${event.affected_project}` : null,
      event.policy_verdict,
      event.policy_rule_id ? `rule: ${event.policy_rule_id}` : null,
      event.policy_reason,
      approval,
      event.feedback_excerpt,
      event.feedback_node_id,
      event.stop_reason,
    ]
      .filter(Boolean);
    item.textContent = parts.join(" · ");
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
      status.textContent = latest.task_status;
    }
    if (list) {
      list.replaceChildren(...events.map(renderEvent));
    }
    if (terminal.has(latest.task_status)) {
      window.clearInterval(interval);
    }
  };

  const interval = window.setInterval(poll, 2000);
  poll();
})();
