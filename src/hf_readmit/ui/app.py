"""Multi-page Streamlit entry point.

Run with: ``streamlit run src/hf_readmit/ui/app.py``

Provides sidebar navigation between the Clinician View and the Monitoring View.
Each page talks to the FastAPI service over HTTP; this entry point only routes.
"""

from __future__ import annotations

import streamlit as st

from hf_readmit.ui import clinician, monitoring

st.set_page_config(page_title="HF Discharge Planning Agent", layout="wide")

PAGES = {
    "Clinician View": clinician.render,
    "Monitoring": monitoring.render,
}


def main() -> None:
    """Route to the selected page."""
    page = st.sidebar.radio("Navigation", list(PAGES))
    PAGES[page]()


main()
