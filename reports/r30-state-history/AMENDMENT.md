# User-authorized operational amendment, 2026-10-09

The user authorized hourly monitoring, engineering repair and continuation, result analysis and evidence-based follow-up experiments, with no cumulative GPU-worker time cap. This supersedes the original no-monitor/no-automatic-follow-up and 36-GPU-hour policy. The scientific R30 matrix, endpoints, seeds, scoring barrier and negative-result retention remain frozen.

New workers enter through `budget_entry.py`, which sets only the effective Guard GPU cap to null. The original resolved configuration, config identity, T0 and cost history remain intact. Existing Python workers finish with their already-loaded guards and are not restarted for this amendment. Per-task/wall deadlines and storage checks remain engineering safeguards. A synthetic guard check confirmed cap removal without mutation of the original configuration or deadlines. See BUDGET_AMENDMENT.json. The original ledger's cap field is historical registration, not the effective cap for amended workers.

Follow-up experiments require a justified hypothesis and frozen protocol; negative outcomes do not authorize selective removal or a claim of success. No new scientific experiment was launched by this operational amendment. A local deployment-script syntax error was corrected before execution and did not affect GPU work.
