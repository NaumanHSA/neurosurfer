"""
DocTemplateGeneratorTool: section-wise documentation generator (RAG + merge)
"""
from __future__ import annotations

import logging
import json, re
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import defaultdict

from neurosurfer.models.chat_models.base import BaseChatModel
from neurosurfer.agents.rag import RAGAgent, pick_files_by_grouped_chunk_hits
from neurosurfer.vectorstores.base import Doc

from ..base_tool import BaseTool, ToolResponse
from ..tool_spec import ToolSpec, ToolParam, ToolReturn


# -----------------------------------------------------------------------------
# Prompts (updated for hybrid template + per-section instructions)
# -----------------------------------------------------------------------------
DOC_GENERATOR_PROMPT = {
    "system_prompt": (
        "You are a section-limited documentation writer.\n"
        "Your job is to write ONLY the body for the requested section of a software project's docs.\n"
        "Hard rules:\n"
        "- Write content for the requested section ONLY. Do NOT reference or preview other sections.\n"
        "- Only return in Markdown with proper headings ('#', '##', etc.). Return structured markdown e.g. (starting with ```markdown)\n"
        "- Use ONLY the provided inputs: folder structure, retrieved context, and section-specific guidance.\n"
        "- Keep it high-level (modules/subsystems, flows, responsibilities). Avoid file-by-file detail.\n"
        "- Code blocks are allowed ONLY for:\n"
        "  • 'Installation / Usage / Implementation Guide' (commands, minimal snippets)\n"
        "  • 'APIs/Endpoints (if applicable)' (1-2 minimal request/response examples)\n"
        "For all other sections, avoid code blocks entirely.\n\n"
        "Specific instructions for this section:\n"
        "{specific_instructions}\n\n"
    ),
    "section_user_prompt": (
        "Write title for section as '{section_title}' and then the body.\n\n"
        "Project folder structure (for clues about modules/architecture):\n"
        "{folder_structure}\n\n"
        "Retrieved context:\n"
        "{context}\n\n"
        "Section-specific guidance (bullets to cover; use them as anchors, not verbatim):\n"
        "{section_instructions}\n\n"
        "Style constraints:\n"
        "- Produce detailed and concise documenation.\n"
        "- Summarize by module/directory (agents, models, db, storage, config), not file-by-file.\n"
        "- No code blocks unless it is necessary.\n"
        "- Write the section title e.g. '# Section Title' followed by the body.\n\n"
        "Specific instructions for this section:\n"
        "{specific_instructions}\n\n"
    ),
}

# ---------------------------------------------------------------------
# Template Planner (LLM-driven): derive section list + per-section notes
# ---------------------------------------------------------------------
DOC_PLANNER_PROMPT = {
    "system_prompt": """You are an expert documentation planner.
Your job is to analyze the provided inputs and produce a best-fit documentation TEMPLATE (not the docs themselves) for the artifact at hand.
The artifact may be a software codebase, a single PDF/article, a research paper, a dataset, a tutorial/guide, a design/architecture doc, a proposal/RFC, a policy/specification, or other written material.

Return STRICTLY valid JSON only (no markdown fences, no commentary):

{
  "sections": [
    {"title": "<section title>", "instructions": "<detailed instructions here>"},
    ...
  ],
  "specific_instructions": "<verbatim + normalized constraints/preferences extracted from the user query>"
}

Intent handling:
- Read the user query carefully and infer intent:
  • If the user explicitly requests specific sections (e.g., “only APIs and Deployment”, “Overview + Use Cases”), output ONLY those sections (no extras), preserving the user's section names and order when possible. If a requested section is not applicable, include the closest sensible alternative and instruct the writer to mark unknown parts as "<to be filled>".
  • If the user requests “complete/full/all documentation” (or equivalent), produce a comprehensive but practical set of sections, (roughly 4-12), tailored to the artifact type.
  • If the user asks a narrow slice (e.g., “installation only”), return a minimal template with just that section.
  • If the user excludes sections (e.g., “everything except Testing”), honor the exclusion.
- Do NOT add sections beyond what the user asked for in targeted mode. In comprehensive mode, choose sections that best fit the artifact.

Specific instructions field:
- Extract any explicit constraints, preferences, or rules stated in the user query (e.g., “Do not expose the actual base URL”, “Use British English”, “No screenshots”, “Cite sources in APA”).
- Return them VERBATIM where sensible and augment with concise, actionable clarifications (e.g., “Mask or generalize any sensitive endpoints; use <to be filled> placeholders where necessary.”).
- If the user did not provide any such constraints, return an empty string "".

Inputs usage:
- Use BOTH inputs when available:
  1) Structure (folder tree, outline, headings) - for coverage hints and terminology.
  2) High-level context - for substance (purpose, architecture/methods, usage, findings, etc.).
- Prefer the high-level context for substance; use structure for coverage without going file-by-file.

Section guidance:
- Each "instructions" must be explicit, detailed, and actionable. Guide the writer on: purpose/aims, scope/audience, key concepts, responsibilities/roles, interactions/flows, methods/procedures, data/models, evaluation/metrics, configuration/parameters, usage/examples, deployment/operational concerns, constraints/assumptions, risks/ethics/compliance, limitations, roadmap/future work, references/attribution, etc., as appropriate to the artifact and user intent.
- If important details are unknown, instruct the writer to use "<to be filled>" or to make and clearly flag reasonable assumptions.
- Avoid code blocks and sample code.
- Output only the JSON object described above, nothing else.
""",
    "user_prompt": """Inputs for planning:

User query (intent; may specify specific sections/inclusions/exclusions, constraints, or "complete"):
{user_query}

Structure (may be empty or not applicable):
{folder_structure}

High-level context (excerpted from relevant files/chunks or the primary document):
{context}

Now produce the documentation TEMPLATE as per the system instructions, in STRICT JSON only (no markdown fences)."""
}

HL_QUERIES_BASE = [
    "Project overview and goals; summarize purpose, problem, and target users",
    "System architecture and high-level design; major components and interactions",
    "Key modules and responsibilities; data flow and control flow",
    "Public APIs/endpoints; main interfaces and contracts",
    "Installation and usage; getting started; quickstart",
    "Configuration and environment variables",
    "Deployment and runtime architecture; CI/CD; scaling",
    "Testing strategy; quality; coverage; linters",
    "Roadmap; limitations; known issues; future work",
]

DOC_LIKE_HINTS = [
    "readme", "overview", "architecture", "design", "adr", "docs", "usage",
    "getting-started", "install", "installation", "api", "endpoints",
    "contributing", "roadmap", "deployment", "config", "testing", "guide"
]
DOC_LIKE_EXTS = {".md", ".rst", ".txt", ".ipynb", ".pdf", ".docx"}


class DocsGenerator(BaseTool):
    """
    Generates a full project documentation by iterating fixed sections and synthesizing each one with RAG context.
    The tool streams a single merged Markdown document (grand doc).

    Purpose:
      Create a high-level documentation template using a fixed section list and the project's folder structure,
      avoiding large single-prompt context windows by generating section-wise.

    Inputs:
      - query (str): Optional short description/instruction for the doc generation run (e.g., style hints).
        All other runtime options come via **kwargs.

    Common kwargs:
      - folder_structure (str): The textual tree of the project directories/files. Recommended.
      - sections (List[str]): Ordered list of section titles to generate. Defaults to the hybrid template.
      - section_instructions (Dict[str, str]): Extra hints per section title.
      - chat_history (List[Dict]): Prior conversation turns if you track them.
      - top_k (int): Retrieval depth (default 10).
      - temperature (float): LLM sampling temperature (default 0.3).
      - max_new_tokens (int): Cap per-section generation (default 700).
      - prepend_title (bool): Whether to start doc with "# Project Documentation" (default True).
      - doc_title (str): Custom H1 title if prepend_title is True (default "Project Documentation").
    """
    spec = ToolSpec(
        name="docs_generator",
        description="Generates structured documentation by first planning sections from the user query, folder structure, "
        "and high-level context, then writing each section using retrieval-augmented generation. "
        "Adapts to intent: can create full documentation or only specific sections requested by the user.",
        when_to_use="Use when the user asks for documentation generation.",
        inputs=[
            ToolParam(name="user_query", type="string", description="user_query (str) for goals, specific instructions, or section constraints.", required=True),
        ],
        returns=ToolReturn(type="string", description="Generated documentation."),
    )

    def __init__(
        self,
        llm: BaseChatModel,
        rag_agent: RAGAgent,
        folder_structure: str = None,
        n_files_per_section: int = 10,
        candidate_pool_size: int = 300,
        max_total_chunks: int = 30,
        file_key: str = "filename",
        max_new_tokens: int = 3000,
        temperature: float = 0.7,
        stream: bool = True,
        logger: Optional[logging.Logger] = logging.getLogger(__name__),
    ):
        self.llm = llm
        self.rag_agent = rag_agent
        self.prompt = DOC_GENERATOR_PROMPT
        self.planner_prompt = DOC_PLANNER_PROMPT
        self.folder_structure: str = folder_structure or ""
        self.n_files_per_section: int = n_files_per_section
        self.candidate_pool_size: int = candidate_pool_size
        self.max_total_chunks: int = max_total_chunks
        self.file_key: str = file_key
        self.max_new_tokens: int = max_new_tokens
        self.temperature: float = temperature
        self.stream: bool = stream
        self.logger = logger

    def __call__(
        self,
        user_query: str,
        **kwargs: Any,
    ) -> ToolResponse:
        documentation_plan = kwargs.get("documentation_plan", {})
        chat_history = kwargs.get("chat_history", [])
        temperature = kwargs.get("temperature", self.temperature)
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        max_total_chunks = kwargs.get("max_total_chunks", self.max_total_chunks)

        self.logger.info("Planning document template...\n")
        if not documentation_plan:
            documentation_plan: Dict[str, Any] = self.plan_docs(
                user_query=user_query,
                folder_structure=self.folder_structure,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                max_total_chunks=max_total_chunks,
                verbose=True
            )
        documentation_plan = documentation_plan.get("sections", {})
        specific_instructions = documentation_plan.get("specific_instructions", "")

        # Build a streaming generator that yields the grand doc
        self.logger.info("Building a streaming generator that yields the grand doc")
        # complete_documentation_markdown = ""
        def _grand_doc_stream() -> Iterable[str]:
            # Iterate through all sections, retrieve context, then generate that section
            for section_title__, section_instructions__ in documentation_plan.items():
                # yield f"## {section_title__}"
                # Step 1: Retrieve context for this section
                retrieval_query = (
                    f"Retrieve information needed to write the '{section_title__}' section of the project documentation. "
                    f"Focus ONLY on content relevant to this section. "
                    f"Section guidance:\n{section_instructions__}\n\n"
                    f"Prioritize high-level details about modules, subsystems, responsibilities, and flows. "
                    f"Avoid unrelated topics or details meant for other sections. "
                )

                # Step 0: Pick files for this section
                files_of_interest = pick_files_by_grouped_chunk_hits(
                    embedder=self.rag_agent.embedder,
                    vector_db=self.rag_agent.vector_db,
                    section_query=retrieval_query,
                    candidate_pool_size=self.candidate_pool_size,
                    n_files=self.n_files_per_section,
                    file_key=self.file_key,
                )
             
                retrieval_results = self.rag_agent.retrieve(
                    user_query=retrieval_query,
                    base_system_prompt=self.prompt["system_prompt"].format(specific_instructions=specific_instructions),
                    base_user_prompt=self.prompt["section_user_prompt"],
                    chat_history=chat_history,
                    top_k=max_total_chunks,
                    metadata_filter={self.file_key: files_of_interest},
                )

                # Step 2: Construct per-section user prompt
                user_prompt = self.prompt["section_user_prompt"].format(
                    section_title=section_title__,
                    folder_structure=(self.folder_structure or "N/A"),
                    context=retrieval_results.context,
                    section_instructions=section_instructions__,
                    specific_instructions=specific_instructions,
                )
                if self.logger:
                    self.logger.info(f"[DocsGeneratorTool] Generating section: {section_title__}")

                # Step 3: Emit a title, and then stream the model output
                for chunk in self.llm.ask(
                    system_prompt=self.prompt["system_prompt"].format(specific_instructions=specific_instructions),
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_new_tokens=getattr(retrieval_results, "max_new_tokens", max_new_tokens),
                    stream=self.stream,
                ):  
                    yield chunk
                yield self.llm._wrap_response(call_id="", model_name="", content="\n\n")
        return ToolResponse(final_answer=True, results=_grand_doc_stream())

    
    def plan_docs(
        self,
        user_query: str,
        folder_structure: str,
        temperature: float = 0.3,
        max_new_tokens: int = 2500,
        verbose: bool = False,
        *,
        extra_hl_query: str = "",
        candidate_pool_size: int = 300,
        n_files: int = 12,
        max_total_chunks: int = 20,
    ) -> Dict[str, str]:
        """
        Returns a mapping: {section_title: instructions}
        """
        # 1) Build high-level context if we can
        high_level_context = ""
        try:
            high_level_context, chosen_files, picked = self.select_high_level_context(
                extra_query=extra_hl_query,
                candidate_pool_size=candidate_pool_size,
                n_files=n_files,
                max_total_chunks=max_total_chunks,
            )
        except Exception as e:
            if self.logger:
                    self.logger.warning(f"[plan_docs] high-level context selection failed: {e}")

        # 2) Ask LLM with refined prompt
        user_prompt = self.planner_prompt["user_prompt"].format(
            user_query=user_query,
            folder_structure=(folder_structure or "").strip(),
            context=(high_level_context or "N/A").strip()
        )
        llm_response = ""
        system_prompt = self.planner_prompt["system_prompt"]
        for chunk in self.llm.ask(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            stream=True,
        ):
            part = chunk["choices"][0]["message"]["content"]
            llm_response += part
            if verbose:
                print(part, end="", flush=True)
        if verbose:
            print("\n\n", flush=True)

        # 3) Robust JSON extraction
        fence = re.search(r"```json\s*(\{.*?\})\s*```", llm_response, flags=re.DOTALL)
        if fence:
            llm_response = fence.group(1)
        else:
            brace = re.search(r"(\{.*\})", llm_response, flags=re.DOTALL)
            if brace:
                llm_response = brace.group(1)

        data = json.loads(llm_response)
        sections_json = data.get("sections", [])

        # 4) Map (title -> instructions) preserving order, dedup titles
        plan_: Dict[str, Any] = {
            "sections": {},
            "specific_instructions": data.get("specific_instructions", "")
        }
        for item in sections_json:
            title = (item.get("title") or "").strip()
            instructions = (item.get("instructions") or "").strip()
            if title and title not in plan_:
                plan_["sections"][title] = instructions or "- <to be filled>"
        return plan_
        
    def build_high_level_queries(self, extra_query: str = "") -> list[str]:
        keywords = []
        # mine top-level directories from the folder tree text
        top_dirs = re.findall(r"^\s*(?:├─|└─)?\s*([A-Za-z0-9_\-\.]+)/\s*$", self.folder_structure, flags=re.MULTILINE)
        top_dirs = [d.lower() for d in top_dirs]
        for k in ["docs", "documentation", "api", "apis", "endpoints", "models", "agents", "services",
                "config", "configs", "deploy", "deployment", "tests", "examples"]:
            if any(k in d for d in top_dirs):
                keywords.append(k)

        tailored = []
        if "docs" in keywords:
            tailored.append("Look for README, overview, architecture, design docs inside /docs")
        if "api" in keywords or "apis" in keywords:
            tailored.append("Project API overview and main endpoints")
        if "deploy" in keywords or "deployment" in keywords:
            tailored.append("Deployment strategy and runtime topology")
        if "config" in keywords or "configs" in keywords:
            tailored.append("Configuration surface and environment variables")

        base = HL_QUERIES_BASE.copy()
        if tailored:
            base.extend(tailored)
        if extra_query:
            base.append(extra_query)
        return base

    def _doc_likelihood_boost(self, meta: Dict[str, Any]) -> float:
        name = (meta.get("file_name") or meta.get("file_path") or "").lower()
        ext  = meta.get("file_ext") or ""
        score = 0.0
        if any(h in name for h in DOC_LIKE_HINTS): score += 0.6
        if ext in DOC_LIKE_EXTS: score += 0.4
        if "/docs/" in name or name.startswith("docs/"): score += 0.3
        return score

    def diverse_pick(
        self,
        hits: List[Tuple[Doc, float]],
        k: int,
        file_key: str = "filename",
        per_file_min: int = 0,
        per_file_cap: int | None = None,
    ) -> List[Tuple[Doc, float]]:
        """
        Two-pass selector:
        1) satisfy per_file_min for as many files as possible
        2) fill remaining by score order, respecting per_file_cap
        """
        out = []
        seen_ids = set()
        counts = defaultdict(int)

        # pass 1: guarantee min-per-file
        if per_file_min > 0:
            for doc, sim in hits:
                if len(out) >= k: break
                if doc.id in seen_ids: continue
                fp = doc.metadata.get(file_key, "")
                if counts[fp] >= per_file_min: continue
                out.append((doc, sim))
                seen_ids.add(doc.id)
                counts[fp] += 1

        # pass 2: fill remainder
        for doc, sim in hits:
            if len(out) >= k: break
            if doc.id in seen_ids: continue
            fp = doc.metadata.get(file_key, "")
            if per_file_cap is not None and counts[fp] >= per_file_cap:
                continue
            out.append((doc, sim))
            seen_ids.add(doc.id)
            counts[fp] += 1

        return out

    def select_high_level_context(
        self,
        extra_query: str = "",
        candidate_pool_size: int = 300,
        n_files: int = 10,
        max_total_chunks: int = 20,
        per_file_min: int = 0,          # e.g., 1 to guarantee coverage across files
        per_file_cap: int | None = None, # e.g., 6 to prevent single-file domination
        overfetch_factor: int = 3,       # safety margin for dedupe/caps
        file_key: str = "filename",
        max_chars_per_chunk: int = 1200,
    ):
        # ---- (1) choose files same as before (you can keep your prior implementation) ----
        queries = self.build_high_level_queries(extra_query=extra_query)
        by_file_sum: Dict[str, float] = defaultdict(float)
        by_file_max: Dict[str, float] = defaultdict(float)
        any_meta: Dict[str, Dict[str, Any]] = {}

        for q in queries:
            qemb = self.rag_retriever_agent.embedder.embed(q)
            hits = self.rag_retriever_agent.vector_db.similarity_search(query_embedding=qemb, top_k=candidate_pool_size)
            for doc, sim in hits:
                fp = doc.metadata.get(file_key)
                if not fp:
                    continue
                any_meta[fp] = doc.metadata
                by_file_sum[fp] += sim
                by_file_max[fp] = max(by_file_max[fp], sim)

        def _doc_boost(meta: Dict[str, Any]) -> float:
            name = (meta.get("file_name") or meta.get("file_path") or "").lower()
            ext  = meta.get("file_ext") or ""
            hints = ["readme", "overview", "architecture", "design", "adr", "docs",
                    "usage", "install", "installation", "api", "endpoints",
                    "contributing", "roadmap", "deployment", "config", "testing", "guide"]
            doc_exts = {".md", ".rst", ".txt", ".ipynb", ".pdf", ".docx"}
            score = 0.0
            if any(h in name for h in hints): score += 0.6
            if ext in doc_exts: score += 0.4
            if "/docs/" in name or name.startswith("docs/"): score += 0.3
            return score

        scored_files = []
        for fp in by_file_sum:
            base = 0.7 * by_file_sum[fp] + 0.3 * by_file_max[fp]
            scored_files.append((fp, base + _doc_boost(any_meta.get(fp, {}))))
        scored_files.sort(key=lambda x: x[1], reverse=True)
        chosen_files = [fp for fp, _ in scored_files[:n_files]] or list(any_meta.keys())[:1]

        # ---- (2) single global retrieval restricted to chosen files ----
        # This is the key change: we no longer cap at chunks_per_file;
        # we pool all restricted hits and then take the top max_total_chunks.
        overview_q = (
            "High-level project overview: purpose, architecture, components, data flow, "
            "APIs/interfaces, setup/usage, configuration, deployment, testing, roadmap"
        )
        qemb = self.rag_retriever_agent.embedder.embed(overview_q)
        fetch_k = max_total_chunks * overfetch_factor
        restricted_hits = self.rag_retriever_agent.vector_db.similarity_search(
            query_embedding=qemb,
            top_k=fetch_k,
            metadata_filter={file_key: chosen_files},
        )

        # ---- (3) pick up to max_total_chunks, optionally with diversity constraints ----
        picked = self.diverse_pick(
            hits=restricted_hits,
            k=max_total_chunks,
            file_key=file_key,
            per_file_min=per_file_min,
            per_file_cap=per_file_cap,
        )

        # ---- (4) build context blocks grouped by file (preserves readability) ----
        group: Dict[str, List[str]] = defaultdict(list)
        for doc, sim in picked:
            fp = doc.metadata.get(file_key, "UNKNOWN")
            text = (doc.text or "").strip()
            if len(text) > max_chars_per_chunk:
                text = text[:max_chars_per_chunk] + "…"
            group[fp].append(text)

        blocks = []
        for fp, texts in group.items():
            blocks.append("### " + fp + "\n" + "\n\n".join(texts))

        high_level_context = "\n\n---\n\n".join(blocks)
        return high_level_context, chosen_files, picked