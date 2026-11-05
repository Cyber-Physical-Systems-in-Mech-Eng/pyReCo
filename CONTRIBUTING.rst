============
Contributing
============

Welcome to the ``pyReCo`` contributor's guide. All users and contributors are expected to be open,
considerate, reasonable and respectful.

Issue reports
=============

If you experience a bug or have a feature request, look at the `issue tracker`_ first (closed issues
included) and open a new issue if you do not find it. Please add your operating system, your Python and
``pyreco`` version, and the steps to reproduce the problem as a minimal example.

Development setup
=================

Fork the repository, clone your fork and install the package in editable mode with the test dependencies::

    python -m pip install -e ".[test]"

Run the tests (the settings, including the coverage report, are in ``pyproject.toml``)::

    python -m pytest

TensorFlow is optional and only needed for the two ``examples/mwe_rc_vs_ann_*.py`` examples
(``pip install -e ".[benchmarks]"``).

Documentation
=============

The documentation uses Sphinx_ with MyST_ (Markdown) and reStructuredText. The module reference in
``docs/api`` is generated from the docstrings when the documentation is built. Build it locally with::

    python -m pip install -r docs/requirements.txt
    sphinx-build -b html docs docs/_build/html

Pull requests
=============

#. Create a branch for your change and keep the change focused.
#. Add or update tests for it, and run the whole test suite.
#. Add a line to ``CHANGELOG.md``; describe changes that alter results or break the API in
   ``UPGRADING.md``.
#. Open a pull request against ``main``. The tests run on every pull request.

Releases (maintainers)
======================

The version number comes from the git tag (hatch-vcs). A release is a tag and a GitHub Release; the
release workflow runs the tests first and then publishes to PyPI. Do not merge unfinished work into
``main``.

.. _issue tracker: https://github.com/Cyber-Physical-Systems-in-Mech-Eng/pyReCo/issues
.. _Sphinx: https://www.sphinx-doc.org/
.. _MyST: https://myst-parser.readthedocs.io/
