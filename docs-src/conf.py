# Sphinx configuration for the WhatAreYouSaying (WAYS) documentation.
#
#   pip install --require-hashes -r docs-src/requirements.txt
#   sh docs-src/build.sh            build, check and pack into docs/site/ (what buildforpeople.dev serves)
#
# Most pages include sections of the README and the files in docs/, so the docs never disagree with them.
# include_read below adjusts what those files link to: images load from raw.githubusercontent.com, the other
# docs/ files become links to their pages here, every other file links to its page on GitHub, and the
# README's reference anchors become targets that work across pages.
import os
import posixpath
import re
from datetime import datetime, timezone

project = "WhatAreYouSaying"
author = "Mr Pirate"
copyright = f"{datetime.now(timezone.utc).year} Mr Pirate, MIT licence"

REPO = "https://github.com/AlveeeRahman/WhatAreYouSaying"
RAW = "https://raw.githubusercontent.com/AlveeeRahman/WhatAreYouSaying/main/"

extensions = ["myst_parser", "sphinx_design", "sphinx_copybutton", "sphinx.ext.mathjax"]
myst_enable_extensions = ["colon_fence", "attrs_block", "attrs_inline", "dollarmath"]
myst_heading_anchors = 3
suppress_warnings = []

exclude_patterns = ["_build", "README.md", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]
html_sidebars = {"**": ["docs-nav.html"]}

# buildforpeople.dev serves this build under /docs/whatareyousaying/ and rewrites every canonical link there.
html_baseurl = os.environ.get("DOCS_BASE_URL", "https://buildforpeople.dev/docs/whatareyousaying/")

html_theme = "pydata_sphinx_theme"
html_title = "WhatAreYouSaying documentation"
html_short_title = "WhatAreYouSaying"
html_logo = "../docs/images/ways_icon.png"
html_favicon = "../docs/images/ways_icon.png"
html_static_path = ["_static"]
html_css_files = ["astryx.css"]
html_js_files = [("glow.js", {"defer": "defer"})]
html_copy_source = False
html_show_sourcelink = False
html_last_updated_fmt = None

html_theme_options = {
    "logo": {"text": "WhatAreYouSaying", "alt_text": "WhatAreYouSaying documentation, home"},
    "navbar_align": "left",
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button-field"],
    "header_links_before_dropdown": 6,
    "icon_links": [{"name": "GitHub", "url": REPO, "icon": "fa-brands fa-github"}],
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "show_prev_next": True,
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "footer_start": ["copyright"],
    "footer_end": [],
    "pygments_light_style": "default",
    "pygments_dark_style": "github-dark",
}
html_context = {
    "github_user": "AlveeeRahman",
    "github_repo": "WhatAreYouSaying",
    "github_version": "main",
    "doc_path": "docs-src",
    "default_mode": "auto",
}

# The files in docs/ that have a page here, as the explicit target at the top of that page.
DOC_PAGES = {
    "docs/tools.md": "doc-tools",
    "docs/backends.md": "doc-backends",
    "docs/reproducibility.md": "doc-reproducibility",
    "docs/reproducibility_audit.md": "doc-reproducibility-audit",
    "docs/help-wanted.md": "doc-help-wanted",
    "CHANGELOG.md": "doc-changelog",
}
IMAGE = re.compile(r"\.(png|jpe?g|gif|svg|webp)$", re.I)
MD_LINK = re.compile(r"(!?)\[((?:[^\[\]]|\[[^\]]*\])*)\]\((?!https?:|#|mailto:)([^)\s]+)\)")
HTML_SRC = re.compile(r'(\s(?:src|href)=")(?!https?:|#|mailto:)([^"]+)"')
REF_ANCHOR = re.compile(r'^<a id="(ref-\d+)"></a>', re.M)


def _target(base, target, image):
    path, _, anchor = target.partition("#")
    full = posixpath.normpath(posixpath.join(base, path)) if path else ""
    if image or IMAGE.search(full):
        return RAW + full
    if full in DOC_PAGES:
        return "#" + DOC_PAGES[full]
    return f"{REPO}/blob/main/{full}" + (f"#{anchor}" if anchor else "")


def include_read(app, relative_path, parent_docname, content):
    rel = os.path.relpath(os.path.join(app.srcdir, str(relative_path)), os.path.join(app.srcdir, ".."))
    base = posixpath.dirname(rel.replace(os.sep, "/"))
    text = content[0]
    text = MD_LINK.sub(lambda m: f"{m[1]}[{m[2]}]({_target(base, m[3], bool(m[1]))})", text)
    text = HTML_SRC.sub(lambda m: f'{m[1]}{_target(base, m[2], m[1].strip().startswith("src"))}"', text)
    text = REF_ANCHOR.sub(lambda m: f"({m[1]})=\n", text)
    content[0] = text


def setup(app):
    app.connect("include-read", include_read)
