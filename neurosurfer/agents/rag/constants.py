from typing import Set

exclude_dirs_in_code: Set[str] = {
    # 🐍 Python
    '.venv', 'venv', '__pycache__', 'unsloth_compiled_cache',
    '.mypy_cache', '.pytest_cache', '.ipynb_checkpoints', '.cache', '.coverage',

    # 🧪 Testing, temp, experiments
    'tmp', 'temp', 'test', 'tests', '__tests__', 'testing', 'sandbox', 'examples', 'samples', 'experiments',

    # 🟨 JavaScript / Node.js
    'node_modules', 'bower_components', 'jspm_packages',

    # ☕ Java
    'target', 'out', '.gradle', '.settings', '.classpath', '.project',

    # 🔷 .NET / C#
    'bin', 'obj', '.vs', '.vscode',

    # 🦀 Rust
    'target',

    # 🐹 Go
    'vendor',

    # 🧊 C/C++
    'build', 'cmake-build-debug', 'cmake-build-release', '.ccls-cache',

    # 🎨 Frontend frameworks
    '.next', 'next', '.nuxt', 'nuxt', 'dist', 'build', 'public', 'static',

    # 🧪 DevOps & CI/CD
    '.circleci', '.github', '.gitlab', '.azure-pipelines', '.husky',

    # 🔄 Version control / IDEs / Configs
    '.git', '.svn', '.hg', '.idea', '.vscode', '.editorconfig',

    # 📦 Containers & envs
    '.docker', '.devcontainer', '.kube', '.kubernetes', 'docker', 'containers', 'k8s',

    # 💻 System-specific & OS metadata
    '.DS_Store', 'Thumbs.db', 'desktop.ini',

    # 📁 Other tooling caches
    '.coverage', '.nyc_output', '.parcel-cache', '.svelte-kit', '.eslintcache', '.turbo',

    # ⚠️ Deprecated or unused project folders
    'archive', 'old', 'legacy', 'deprecated', 'trash'
}



supported_file_types: Set[str] = {
    # ----------------------------------------------------
    # General text formats
    # ----------------------------------------------------
    ".txt", ".text", ".ascii",

    # ----------------------------------------------------
    # Rich Text / Markup / Documents
    # ----------------------------------------------------
    ".pdf", ".docx", ".doc", ".odt", ".rtf",
    ".html", ".htm", ".xhtml", ".xml",
    ".md", ".markdown", ".mdx",
    ".rst", ".tex", ".latex",

    # ----------------------------------------------------
    # Data / tabular formats
    # ----------------------------------------------------
    ".csv", ".tsv", ".tab",
    ".xls", ".xlsx", ".xlsm", ".ods",
    ".json", ".jsonl",
    ".yaml", ".yml",

    # ----------------------------------------------------
    # Config & environment
    # ----------------------------------------------------
    ".env", ".ini", ".cfg", ".conf", ".config", ".properties",
    ".toml",

    # ----------------------------------------------------
    # Logs
    # ----------------------------------------------------
    ".log", ".out", ".err",

    # ----------------------------------------------------
    # Code files (common languages)
    # ----------------------------------------------------
    ".py", ".pyi",
    ".java", ".kt", ".kts",
    ".js", ".jsx", ".cjs", ".mjs",
    ".ts", ".tsx",
    ".cpp", ".cc", ".cxx", ".hpp", ".h", ".c",
    ".cs",
    ".go",
    ".rb",
    ".rs",
    ".php",
    ".swift",
    ".scala",
    ".lua",
    ".r",
    ".sh", ".bash", ".zsh",
    ".ps1", ".psm1",
    ".bat", ".cmd",
    ".sql",
    ".dockerfile", ".dockerignore",
    ".makefile", "makefile",
    ".gradle", ".pom", ".sbt",

    # ----------------------------------------------------
    # Web templates / frontend text
    # ----------------------------------------------------
    ".css", ".scss", ".sass", ".less",
    ".vue", ".svelte",
    ".handlebars", ".hbs", ".mustache",
    ".ejs", ".twig", ".jinja", ".jinja2",

    # ----------------------------------------------------
    # Notebooks (extract markdown + code)
    # ----------------------------------------------------
    ".ipynb",

    # ----------------------------------------------------
    # Graph / DSL formats
    # ----------------------------------------------------
    ".dot", ".gv",        # Graphviz
    ".plantuml", ".puml", # PlantUML diagrams
    ".mermaid",           # Mermaid diagrams

    # ----------------------------------------------------
    # Subtitles / annotations
    # ----------------------------------------------------
    ".srt", ".vtt",

    # ----------------------------------------------------
    # Documentation datasets
    # ----------------------------------------------------
    ".bib",       # BibTeX
    ".enw",       # EndNote
    ".ris",       # RIS reference format

    # ----------------------------------------------------
    # Misc structured text
    # ----------------------------------------------------
    ".csv.gz", ".txt.gz",   # Compressed text data
    ".ndjson",              # Newline-delimited JSON
}
