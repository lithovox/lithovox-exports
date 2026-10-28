`xslope_input_template.xlsx` is XSLOPE's blank input template, template
version 26, as released in xslope 0.5.2 on PyPI
(`xslope/resources/input_template.xlsx`; https://github.com/njones61/xslope),
© Norman L. Jones, Apache License 2.0
(https://www.apache.org/licenses/LICENSE-2.0). xslope.py fills a copy of it;
the file itself is unchanged.

Take it from a *released* xslope, not from GitHub's main branch: XSLOPE refuses
files of a newer template version than it knows, so a file made from an
unreleased template can't be opened by the XSLOPE users have. To follow a new
release, copy its template here, publish a new lithovox-exports, and run the
API's tests/corridor/test_xslope.py (after `uv add --dev xslope@latest` there),
which reads the files with xslope's own loader.
