import os
import json
import logging
from typing import Optional
from openai import OpenAI
from env.models import ObservationModel, ActionModel, AmbulanceState, Severity

# Standard logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PriorityAgent:
    """
    Production-ready LLM-based agent for ambulance dispatch.
    Uses OpenRouter API with heuristic fallbacks.
    """
    def __init__(self):
        # 2. ADD FALLBACK .env SUPPORT
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except:
            pass

        # 1. ADD MULTI-SOURCE API KEY LOADING
        api_key = (
            os.getenv("HF_TOKEN") or 
            os.getenv("OPENAI_API_KEY") or 
            os.environ.get("HF_TOKEN") or 
            os.environ.get("OPENAI_API_KEY")
        )

        force_heuristic = os.getenv("FORCE_HEURISTIC", "false").lower() == "true"

        # 3. REMOVE HARD CRASH & 5. ENSURE SYSTEM DOES NOT CRASH
        if force_heuristic or not api_key:
            if force_heuristic:
                print("INFO: Heuristic mode FORCED via environment variable.")
            else:
                print("WARNING: API key not set, using fallback agent")
            self.use_fallback = True
            self.client = None
        else:
            self.use_fallback = False
            self.api_key = api_key
            self.base_url = "https://openrouter.ai/api/v1"
            self.model = os.getenv("MODEL_NAME") or "openai/gpt-4o-mini"
            
            # 4. CONDITIONAL CLIENT INIT
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

    def _heuristic_fallback(self, observation: ObservationModel) -> ActionModel:
        """Heuristic logic from SmartDispatchAgent to ensure robustness."""
        idle_ambs = [a for a in observation.ambulances if a.state == "idle"]
        if not idle_ambs or not observation.emergencies:
            return ActionModel(ambulance_id=None, emergency_id="", hospital_id=None)

        priority_map = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.NORMAL: 2}
        open_emgs = [e for e in observation.emergencies if not e.assigned]
        
        if not open_emgs:
            return ActionModel(ambulance_id=None, emergency_id="", hospital_id=None)

        sorted_emgs = sorted(
            open_emgs,
            key=lambda e: (priority_map.get(e.severity, 3), e.time_remaining)
        )
        selected_emg = sorted_emgs[0]
        best_amb = min(idle_ambs, key=lambda a: abs(a.node - selected_emg.node))
        
        available_hosps = [h for h in observation.hospitals if h.current_patients < h.capacity]
        if not available_hosps:
            return ActionModel(ambulance_id=None, emergency_id="", hospital_id=None)

        selected_hosp = min(available_hosps, key=lambda h: abs(h.node - selected_emg.node))
        
        return ActionModel(
            ambulance_id=best_amb.id,
            emergency_id=selected_emg.id,
            hospital_id=selected_hosp.id
        )

    def act(self, observation: ObservationModel) -> ActionModel:
        """Core dispatch logic with LLM decision making and fallback safety."""
        # 5. ENSURE SYSTEM DOES NOT CRASH
        if self.use_fallback or self.client is None:
            return self._heuristic_fallback(observation)

        # 3. IMPROVE SYSTEM PROMPT (VERY IMPORTANT)
        system_prompt = (
            "You are an ambulance dispatch AI.\n\n"
            "Your job:\n"
            "- Always assign the nearest available ambulance to the highest priority emergency.\n"
            "- Always choose a valid emergency_id from the observation.\n"
            "- Never return empty or null actions unless absolutely no ambulances or emergencies exist.\n\n"
            "Rules:\n"
            "- ambulance_id must be a valid integer\n"
            "- emergency_id must be a valid string from input\n"
            "- hospital_id must be a valid integer\n\n"
            "If emergencies exist:\n"
            "YOU MUST assign an ambulance.\n\n"
            "Return ONLY JSON:\n"
            "{\n"
            "  \"ambulance_id\": int,\n"
            "  \"emergency_id\": string,\n"
            "  \"hospital_id\": int\n"
            "}"
        )
        
        obs_dict = observation.model_dump() if hasattr(observation, "model_dump") else observation.dict()
        observation_json = json.dumps(obs_dict)
        
        if len(observation_json) > 15000:
            observation_json = observation_json[:15000] + "... [truncated]"

        try:
            # 1. REDUCE TOKEN USAGE (CRITICAL) & 2. ADD TEMPERATURE CONTROL
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": observation_json}
                ],
                temperature=0.2,
                max_tokens=300,
                timeout=15.0
            )
            
            # 1. CLEAN LLM OUTPUT (CRITICAL FIX)
            raw_output = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks
            if raw_output.startswith("```"):
                raw_output = raw_output.replace("```json", "")
                raw_output = raw_output.replace("```", "")
                raw_output = raw_output.strip()
            
            # 2. SAFE JSON PARSING
            try:
                parsed = json.loads(raw_output)
            except:
                parsed = {}
            
            # 3. VALIDATE OUTPUT FIELDS
            amb_id = parsed.get("ambulance_id")
            emg_id = parsed.get("emergency_id")
            hosp_id = parsed.get("hospital_id")
            
            # 4. FIX INVALID VALUES
            if emg_id is None:
                emg_id = ""
            
            if amb_id is None:
                amb_id = None
                
            if hosp_id is None:
                hosp_id = None

            # 4. ADD FALLBACK IF EMPTY RESPONSE
            # If the LLM returns null/empty but we have available work, use heuristic
            if (amb_id is None or emg_id == "") and any(not e.assigned for e in observation.emergencies):
                return self._heuristic_fallback(observation)
            
            # 5. ADD DEBUG PRINT
            logging.debug("FINAL ACTION: %s %s %s", amb_id, emg_id, hosp_id)

            # Validate IDs exist in observation
            amb_exists = any(a.id == amb_id for a in observation.ambulances) if amb_id is not None else False
            emg_exists = any(e.id == emg_id for e in observation.emergencies) if emg_id != "" else False
            
            if amb_exists and emg_exists:
                return ActionModel(
                    ambulance_id=amb_id,
                    emergency_id=emg_id,
                    hospital_id=hosp_id
                )
            
        except Exception as e:
            logger.error(f"LLM Agent failed: {e}. Falling back to heuristic.")
            
        # 6. RETURN ACTION (via Fallback)
        return self._heuristic_fallback(observation)
