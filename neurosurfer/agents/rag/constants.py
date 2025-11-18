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
    # General text and document files
    ".txt", ".pdf", ".html", ".htm", ".docx", ".doc", ".odt",
    
    # Spreadsheets and data files
    ".csv", ".xls", ".xlsx", ".tsv", ".ods", ".json", ".xml", ".yaml", ".yml",

    # Presentations
    ".ppt", ".pptx", ".odp",

    # Code files
    ".py", ".java", ".js", ".ts", ".jsx", ".tsx",
    ".cpp", ".c", ".h", ".cs", ".go", ".rb", ".rs", ".php", ".swift", ".kt",
    ".sh", ".bat", ".ps1", ".scala", ".lua", ".r",

    # Config and markup
    ".env", ".ini", ".toml", ".cfg", ".conf", ".properties",

    # Logs
    ".log",

    # Markdown / Rich text
    ".rst", ".rtf",

    # Misc text files
    ".tex", ".srt", ".vtt"
}