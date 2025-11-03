# Built-in Tools

Neurosurfer ships with a growing set of **ready-to-use tools** built on the same Tooling API as your custom tools. Drop them into a `Toolkit`, compose them in agents, and replace them with your own implementations when needed.

<div class="grid cards" markdown>

-   :material-database-cog:{ .lg .middle } **SQL Tools**

    ---

    Composable building blocks for natural‑language analytics and database understanding: pick relevant tables, generate queries, execute them, and present results.

    [:octicons-arrow-right-24: Explore SQL Tools](./sql/index.md)

</div>


> Tip: Use `Toolkit.get_tools_description()` to emit a human/LLM‑friendly summary of everything you registered.