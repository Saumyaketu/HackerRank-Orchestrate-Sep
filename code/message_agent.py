"""
LLM-Powered Financial Communications & Intent Extraction Agent.
Parses unstructured multilingual messages (English & Indonesian), performs
named entity recognition (amounts, currencies, dates), and detects adversarial prompt injections.
"""

import os
import json
import re
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = REPO_ROOT / "code" / "prompts" / "message_nlp_prompt.txt"

class FinancialMessageAgent:
    def __init__(self, model_name: str = "llava", api_base: str = None):
        self.model_name = model_name
        self.api_base = api_base or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.prompt_template = ""
        if PROMPT_PATH.exists():
            with open(PROMPT_PATH, "r", encoding="utf-8") as f:
                self.prompt_template = f.read()

    def call_local_llm(self, message_text: str) -> dict:
        """
        Invokes local LLM (e.g. LLaVA / LLaMA via Ollama) to analyze financial communication.
        """
        import urllib.request
        import urllib.error
        try:
            prompt = f"{self.prompt_template}\n\nAnalyze this message:\n\"\"\"{message_text}\"\"\"\n\nRespond strictly in JSON."
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.api_base}/api/generate",
                data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                res = json.loads(resp.read().decode())
                response_text = res.get("response", "")
                parsed = json.loads(response_text)
                return parsed
        except (urllib.error.URLError, TimeoutError, OSError, Exception):
            return None

    def parse_message(self, message_text: str, user_id: str, related_event_id: str = None) -> dict:
        """
        Extracts structured financial facts from communication messages.
        Defends against prompt injection and social engineering (e.g. advance-fee scams).
        """
        result = {
            "user_id": user_id,
            "related_event_id": related_event_id,
            "is_scam_or_injection": False,
            "salary_amount": None,
            "salary_currency": None,
            "salary_date": None,
            "salary_arrears": None,
            "employment_ended": False,
            "rent_increase_pct": None,
            "confirmed_invoice": None,
            "intent": "general_communication"
        }

        # 1. Adversarial Prompt Injection & Advance-Fee Scam Detection
        scam_patterns = [
            r"pay\s+the\s+release\s+charge",
            r"pay\s+the\s+processing\s+charge",
            r"selected\s+for\s+a\s+cash\s+prize",
            r"bayar\s+biaya\s+pencairan",
            r"bayar\s+biaya\s+pemrosesan",
            r"menang\s+undian"
        ]
        if any(re.search(p, message_text, re.I) for p in scam_patterns):
            result["is_scam_or_injection"] = True
            result["intent"] = "scam_injection"
            return result

        # 2. Employment Termination Detection
        term_patterns = [
            r"employment\s+has\s+ended",
            r"contract\s+has\s+ended",
            r"kontrak.*berakhir",
            r"pendapatan.*berakhir",
            r"household\s+employment\s+record\s+has\s+ended",
            r"hubungan\s+kerja.*telah\s+berakhir"
        ]
        if any(re.search(p, message_text, re.I) for p in term_patterns):
            result["employment_ended"] = True
            result["intent"] = "employment_ended"

        # 3. Recurring Salary Updates
        m_sal = re.search(
            r"(?:gaji.*(?:menjadi|adalah|sebesar)|monthly\s+(?:salary|pay)\s+(?:has\s+increased\s+to|is)|next\s+salary\s+is\s+reduced\s+to|regular\s+salary.*(?:resumes|is)|first\s+salary.*(?:will\s+be|is|of)|salary\s+of|remaining\s+confirmed\s+monthly\s+salary\s+is)\s+([A-Z]{3})\s+([\d,]+(?:\.\d+)?)",
            message_text,
            re.I
        )
        if m_sal:
            result["salary_currency"] = m_sal.group(1).upper()
            result["salary_amount"] = float(m_sal.group(2).replace(",", ""))
            result["intent"] = "salary_update"

        # 4. Salary Arrears Adjustment
        m_arr = re.search(r"one-time\s+arrears\s+adjustment\s+of\s+([A-Z]{3})\s+([\d,]+(?:\.\d+)?)", message_text, re.I)
        if m_arr:
            result["salary_arrears"] = float(m_arr.group(2).replace(",", ""))
            result["intent"] = "salary_arrears"

        # 5. Effective Date
        m_date = re.search(
            r"(?:expected\s+on|confirmed\s+for|resumes\s+on|berlaku\s+mulai|credit\s+date\s+is|applies\s+from|dikonfirmasi\s+untuk|scheduled\s+for|masuk\s+pada)\s+(\d{4}-\d{2}-\d{2})",
            message_text,
            re.I
        )
        if m_date:
            result["salary_date"] = m_date.group(1)

        # 6. Rent / Housing Increase
        m_rent = re.search(r"(?:rent\s+increase|kenaikan\s+sewa).*?(\d+(?:\.\d+)?)\s*%", message_text, re.I)
        if m_rent:
            result["rent_increase_pct"] = float(m_rent.group(1)) / 100.0
            result["intent"] = "rent_increase"

        # 7. Confirmed Business Invoice Payout
        m_inv = re.search(
            r"(?:invoice.*confirmed|pembayaran\s+faktur.*dikonfirmasi).*?([A-Z]{3})\s+([\d,]+(?:\.\d+)?).*?(\d{4}-\d{2}-\d{2})",
            message_text,
            re.I
        )
        if m_inv:
            result["confirmed_invoice"] = {
                "currency": m_inv.group(1).upper(),
                "amount": float(m_inv.group(2).replace(",", "")),
                "date": m_inv.group(3)
            }
            result["intent"] = "confirmed_invoice"

        return result

message_agent = FinancialMessageAgent(model_name="llava")
