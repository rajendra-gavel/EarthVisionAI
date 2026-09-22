import json
import re

import requests


class AssistantService:

    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/chat"
        self.model = "qwen3:4b"


    # =========================================================
    # MAIN ASSISTANT ROUTER
    # =========================================================

    def answer(
        self,
        question: str,
        change_result=None,
        object_result=None,
    ):

        question = question.strip()

        if not question:
            raise ValueError("Question cannot be empty.")

        question_lower = question.lower()


        # =====================================================
        # 1. COLOR + OBJECT QUESTIONS
        # =====================================================

        if self._is_color_object_question(question):

            answer = self._answer_color_object_question(
                question,
                object_result,
            )

            if answer:
                return answer


        # =====================================================
        # 2. COLOR / IMAGE QUESTIONS
        # =====================================================

        if self._is_color_question(question):

            answer = self._answer_color_question(
                object_result
            )

            if answer:
                return answer


        # =====================================================
        # 3. OBJECT DETECTION QUESTIONS
        # =====================================================

        if self._is_object_question(question):

            answer = self._answer_object_question(
                object_result
            )

            if answer:
                return answer


        # =====================================================
        # 4. CHANGE / DIFFERENCE QUESTIONS
        # =====================================================

        if self._is_change_question(question):

            answer = self._answer_change_question(
                change_result
            )

            if answer:
                return answer


        # =====================================================
        # 5. REGION / SEVERITY QUESTIONS
        # =====================================================

        if self._is_region_question(question):

            answer = self._answer_region_question(
                change_result
            )

            if answer:
                return answer


        # =====================================================
        # 6. ANALYSIS METHOD QUESTIONS
        # =====================================================

        if self._is_method_question(question):

            return self._answer_method_question(
                change_result
            )


        # =====================================================
        # 7. GENERAL QWEN3 EXPLANATION
        # =====================================================

        return self._ask_qwen(
            question,
            change_result,
            object_result,
        )


    # =========================================================
    # OBJECT QUESTION DETECTION
    # =========================================================

    def _is_object_question(
        self,
        question: str,
    ) -> bool:

        q = question.lower()

        terms = [
            "object",
            "objects",
            "feature",
            "features",
            "detect",
            "detected",
            "detection",
            "roundabout",
            "roundabouts",
            "vehicle",
            "vehicles",
            "plane",
            "planes",
            "ship",
            "ships",
            "building",
            "buildings",
        ]

        return any(term in q for term in terms)


    # =========================================================
    # OBJECT ANSWER
    # =========================================================

    def _answer_object_question(
        self,
        object_result,
    ):

        if not object_result:

            return {
                "answer": (
                    "No object-detection result is "
                    "currently available."
                ),
                "model": self.model,
                "source": "earthvision_object_detection",
            }


        counts = object_result.get(
            "counts",
            {},
        )


        if not counts:

            return {
                "answer": (
                    "EarthVision did not detect any "
                    "supported objects or geographic features."
                ),
                "model": object_result.get(
                    "model",
                    self.model,
                ),
                "source": "earthvision_object_detection",
            }


        parts = []

        for name, count in counts.items():

            display_name = str(name)

            if count == 1:
                parts.append(
                    f"1 {display_name}"
                )
            else:
                parts.append(
                    f"{count} {display_name}"
                )


        return {
            "answer": (
                "EarthVision detected "
                + ", ".join(parts)
                + "."
            ),
            "model": object_result.get(
                "model",
                self.model,
            ),
            "source": "earthvision_object_detection",
        }


    # =========================================================
    # CHANGE QUESTION DETECTION
    # =========================================================

    def _is_change_question(
        self,
        question: str,
    ) -> bool:

        q = question.lower()

        terms = [
            "change",
            "changes",
            "changed",
            "difference",
            "differences",
            "different",
            "visual difference",
            "potential difference",
            "potential differences",
            "change area",
            "change percentage",
            "how much changed",
        ]

        return any(term in q for term in terms)


    # =========================================================
    # CHANGE ANSWER
    # =========================================================

    def _answer_change_question(
        self,
        change_result,
    ):

        if not change_result:

            return {
                "answer": (
                    "No satellite change-analysis result "
                    "is currently available."
                ),
                "model": self.model,
                "source": "earthvision_change_analysis",
            }


        percentage = change_result.get(
            "change_percentage"
        )

        regions = change_result.get(
            "regions"
        )

        candidate_regions = change_result.get(
            "candidate_regions"
        )


        if regions is not None:

            region_count = len(regions)

        elif candidate_regions is not None:

            region_count = candidate_regions

        else:

            region_count = None


        if percentage is not None and region_count is not None:

            answer = (
                f"EarthVision identified approximately "
                f"{float(percentage):.2f}% of the image as "
                f"potential visual difference, across "
                f"{region_count} candidate regions. "
                f"These are potential differences from the "
                f"experimental analysis and should not be "
                f"interpreted as confirmed geographic change."
            )

        elif percentage is not None:

            answer = (
                f"EarthVision flagged approximately "
                f"{float(percentage):.2f}% of the image "
                f"as potential visual difference. "
                f"This is experimental analysis and does "
                f"not represent confirmed geographic change."
            )

        elif region_count is not None:

            answer = (
                f"EarthVision identified {region_count} "
                f"candidate regions of potential visual "
                f"difference. These should not be interpreted "
                f"as confirmed geographic change."
            )

        else:

            answer = (
                "EarthVision has identified potential visual "
                "differences using the experimental "
                "change-analysis pipeline."
            )


        return {
            "answer": answer,
            "model": self.model,
            "source": "earthvision_change_analysis",
        }


    # =========================================================
    # REGION QUESTION DETECTION
    # =========================================================

    def _is_region_question(
        self,
        question: str,
    ) -> bool:

        q = question.lower()

        terms = [
            "region",
            "regions",
            "candidate region",
            "candidate regions",
            "severity",
            "high severity",
            "medium severity",
            "low severity",
        ]

        return any(term in q for term in terms)


    # =========================================================
    # REGION ANSWER
    # =========================================================

    def _answer_region_question(
        self,
        change_result,
    ):

        if not change_result:

            return {
                "answer": (
                    "No change-analysis result is "
                    "currently available."
                ),
                "model": self.model,
                "source": "earthvision_change_analysis",
            }


        regions = change_result.get(
            "regions",
            [],
        )


        if not regions:

            candidate_regions = change_result.get(
                "candidate_regions"
            )

            if candidate_regions:

                return {
                    "answer": (
                        f"EarthVision identified "
                        f"{candidate_regions} candidate regions, "
                        f"but detailed region severity information "
                        f"is not available."
                    ),
                    "model": self.model,
                    "source": "earthvision_change_analysis",
                }


            return {
                "answer": (
                    "No detailed candidate-region information "
                    "is currently available."
                ),
                "model": self.model,
                "source": "earthvision_change_analysis",
            }


        question_lower = ""


        severity_counts = {
            "high": 0,
            "medium": 0,
            "low": 0,
        }


        for region in regions:

            severity = str(
                region.get(
                    "severity",
                    "low",
                )
            ).lower()

            if severity in severity_counts:

                severity_counts[severity] += 1


        parts = []

        if severity_counts["high"]:

            parts.append(
                f"{severity_counts['high']} high"
            )

        if severity_counts["medium"]:

            parts.append(
                f"{severity_counts['medium']} medium"
            )

        if severity_counts["low"]:

            parts.append(
                f"{severity_counts['low']} low"
            )


        if parts:

            severity_text = ", ".join(parts)

            answer = (
                f"EarthVision identified "
                f"{len(regions)} candidate regions: "
                f"{severity_text} severity. "
                f"Severity represents the strength of the "
                f"analysis signal and does not confirm "
                f"real-world geographic change."
            )

        else:

            answer = (
                f"EarthVision identified "
                f"{len(regions)} candidate regions. "
                f"Detailed severity information is not available."
            )


        return {
            "answer": answer,
            "model": self.model,
            "source": "earthvision_change_analysis",
        }


    # =========================================================
    # METHOD QUESTION DETECTION
    # =========================================================

    def _is_method_question(
        self,
        question: str,
    ) -> bool:

        q = question.lower()

        terms = [
            "analysis method",
            "method",
            "how does the analysis work",
            "how does change analysis work",
            "how is change detected",
            "how are changes detected",
            "explain the analysis",
            "explain the method",
        ]

        return any(term in q for term in terms)


    # =========================================================
    # METHOD ANSWER
    # =========================================================

    def _answer_method_question(
        self,
        change_result,
    ):

        method = None

        if change_result:

            method = (
                change_result.get(
                    "method_label"
                )
                or change_result.get(
                    "method"
                )
            )


        if method:

            method_text = str(method)

        else:

            method_text = (
                "feature registration, structural "
                "comparison, and spectral comparison"
            )


        answer = (
            "EarthVision uses "
            f"{method_text}. "
            "The two images are first aligned so that "
            "corresponding areas can be compared. Structural "
            "and spectral differences are then analyzed to "
            "identify candidate regions of potential visual "
            "difference. The result is experimental and may "
            "include illumination, seasonal, acquisition, "
            "or residual registration effects."
        )


        return {
            "answer": answer,
            "model": self.model,
            "source": "earthvision_change_analysis",
        }


    # =========================================================
    # COLOR QUESTION DETECTION
    # =========================================================

    def _is_color_question(
        self,
        question: str,
    ) -> bool:

        q = question.lower()

        color_terms = [
            "color",
            "colour",
            "colors",
            "colours",
        ]

        return any(
            term in q
            for term in color_terms
        )


    # =========================================================
    # COLOR + OBJECT QUESTION DETECTION
    # =========================================================

    def _is_color_object_question(
        self,
        question: str,
    ) -> bool:

        q = question.lower()

        colors = [
            "red",
            "orange",
            "yellow",
            "green",
            "cyan",
            "blue",
            "purple",
            "white",
            "gray",
            "grey",
            "black",
        ]

        objects = [
            "object",
            "objects",
            "vehicle",
            "car",
            "bus",
            "truck",
            "person",
            "plane",
            "ship",
            "feature",
        ]

        return (
            any(color in q for color in colors)
            and any(obj in q for obj in objects)
        )


    # =========================================================
    # COLOR + OBJECT ANSWER
    # =========================================================

    def _answer_color_object_question(
        self,
        question: str,
        object_result,
    ):

        if not object_result:

            return {
                "answer": (
                    "No object-detection result is "
                    "currently available."
                ),
                "model": self.model,
                "source": "earthvision_object_detection",
            }


        detections = object_result.get(
            "detections",
            [],
        )


        if not detections:

            return {
                "answer": (
                    "No detected objects are available "
                    "to answer the color-specific question."
                ),
                "model": object_result.get(
                    "model",
                    self.model,
                ),
                "source": "earthvision_object_detection",
            }


        q = question.lower()


        color_map = {
            "red": "red",
            "orange": "orange",
            "yellow": "yellow",
            "green": "green",
            "cyan": "cyan",
            "blue": "blue",
            "purple": "purple",
            "white": "white",
            "gray": "gray",
            "grey": "gray",
            "black": "black",
        }


        requested_color = None

        for word, normalized in color_map.items():

            if re.search(
                rf"\b{re.escape(word)}\b",
                q,
            ):

                requested_color = normalized
                break


        if not requested_color:

            return None


        matches = []


        for detection in detections:

            dominant_color = str(
                detection.get(
                    "dominant_color",
                    ""
                )
            ).lower()


            if dominant_color == requested_color:

                matches.append(detection)


        if not matches:

            return {
                "answer": (
                    f"No detected object was identified "
                    f"with a dominant {requested_color} color."
                ),
                "model": object_result.get(
                    "model",
                    self.model,
                ),
                "source": "earthvision_object_detection",
            }


        if len(matches) == 1:

            detection = matches[0]

            object_name = detection.get(
                "class",
                "object",
            )

            confidence = detection.get(
                "confidence"
            )


            if confidence is not None:

                confidence_text = (
                    f" with {float(confidence) * 100:.1f}% "
                    f"confidence"
                )

            else:

                confidence_text = ""


            answer = (
                f"The {requested_color} object appears "
                f"to be a {object_name}"
                f"{confidence_text}."
            )

        else:

            names = [
                str(
                    item.get(
                        "class",
                        "object",
                    )
                )
                for item in matches
            ]

            answer = (
                f"EarthVision identified {len(matches)} "
                f"objects with a dominant {requested_color} "
                f"color: "
                + ", ".join(names)
                + "."
            )


        return {
            "answer": answer,
            "model": object_result.get(
                "model",
                self.model,
            ),
            "source": "earthvision_object_detection",
        }


    # =========================================================
    # IMAGE COLOR ANSWER
    # =========================================================

    def _answer_color_question(
        self,
        object_result,
    ):

        if not object_result:
            return None


        image_analysis = object_result.get(
            "image_analysis"
        )


        if not image_analysis:
            return None


        colors = image_analysis.get(
            "dominant_colors",
            [],
        )


        if not colors:

            return {
                "answer": (
                    "I could not determine the dominant "
                    "colors from the image."
                ),
                "model": self.model,
                "source": "earthvision_image_analysis",
            }


        names = []

        for item in colors[:5]:

            if isinstance(item, dict):

                name = item.get("name")

                if name:
                    names.append(str(name))

            elif item:

                names.append(str(item))


        if not names:

            return {
                "answer": (
                    "I could not determine the dominant "
                    "colors from the image."
                ),
                "model": self.model,
                "source": "earthvision_image_analysis",
            }


        if len(names) == 1:

            color_text = names[0]

        elif len(names) == 2:

            color_text = (
                f"{names[0]} and {names[1]}"
            )

        else:

            color_text = (
                ", ".join(names[:-1])
                + f", and {names[-1]}"
            )


        brightness = image_analysis.get(
            "brightness_label"
        )


        if brightness:

            answer = (
                f"The dominant colors are "
                f"{color_text}. "
                f"The image appears "
                f"{brightness}."
            )

        else:

            answer = (
                f"The dominant colors are "
                f"{color_text}."
            )


        return {
            "answer": answer,
            "model": self.model,
            "source": "earthvision_image_analysis",
        }


    # =========================================================
    # QWEN3
    # =========================================================

    def _ask_qwen(
        self,
        question,
        change_result,
        object_result,
    ):

        context = {
            "change_analysis": change_result,
            "object_detection": object_result,
        }


        context_json = json.dumps(
            context,
            indent=2,
            default=str,
        )


        system_prompt = """
You are EarthVision AI Assistant.

Answer the user's question using only the supplied
EarthVision analysis data.

IMPORTANT RULES:

1. Give only the final user-facing answer.
2. Never describe your reasoning process.
3. Never mention prompts, context, instructions,
   JSON, system messages, or internal processing.
4. Never say "Let me analyze", "We are given",
   "the context says", or similar meta-language.
5. Do not expose or reproduce the supplied JSON.
6. Do not invent objects, measurements, colors,
   locations, or geographic changes.
7. Keep the answer concise, normally 1 to 3 sentences.
8. If the data is insufficient, simply say that
   the information is not available.
9. Treat experimental change-analysis results as
   potential visual differences, not confirmed
   geographic changes.
"""


        user_prompt = f"""
EarthVision data:

{context_json}

User question:

{question}

Return only a concise answer to the user.
"""


        payload = {
            "model": self.model,

            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],

            "stream": False,

            "think": False,

            "options": {
                "temperature": 0.1,
                "num_predict": 100,
            },
        }


        try:

            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=180,
            )

            response.raise_for_status()

        except requests.exceptions.ConnectionError:

            raise RuntimeError(
                "EarthVision AI Assistant is unavailable. "
                "Please make sure Ollama is running."
            )

        except requests.exceptions.Timeout:

            raise RuntimeError(
                "EarthVision AI Assistant timed out."
            )

        except requests.exceptions.RequestException as exc:

            raise RuntimeError(
                f"Ollama request failed: {exc}"
            )


        data = response.json()


        message = data.get(
            "message",
            {},
        )


        answer = message.get(
            "content",
            "",
        ).strip()


        # =====================================================
        # REMOVE THINKING CONTENT
        # =====================================================

        if "</think>" in answer:

            answer = answer.split(
                "</think>",
                1,
            )[1].strip()


        if "<think>" in answer:

            answer = re.sub(
                r"<think>.*?</think>",
                "",
                answer,
                flags=re.DOTALL,
            ).strip()


        # =====================================================
        # REMOVE COMMON META PREFIXES
        # =====================================================

        prefixes = [
            "Let me analyze the EarthVision data.",
            "Let me analyze the EarthVision analysis data.",
            "We are given the EarthVision analysis context.",
            "Based on the provided context,",
            "According to the provided context,",
        ]


        for prefix in prefixes:

            if answer.lower().startswith(
                prefix.lower()
            ):

                answer = answer[
                    len(prefix):
                ].strip()


        if not answer:

            raise RuntimeError(
                "Ollama returned an empty response."
            )


        return {
            "answer": answer,
            "model": self.model,
            "source": "qwen3",
        }