from langchain_core.output_parsers import BaseOutputParser
import re

class ParaphraseOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        if "<think>" in text:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        quoted = re.findall(r'"(.*?)"', text)
        if quoted:
            return quoted[-1].strip()

        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        if lines and len(lines[-1].split()) >= 3:
            return lines[-1].strip().strip('"')  

        raise ValueError(f"Failed to parse a paraphrased sentence from:\n{text}")


class TextClassificationOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        if "<think>" in text:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        sentiments = {"positive", "negative", "neutral"}
        detected = {s for s in sentiments if s in text.lower()}

        if len(detected) == 1:
            return detected.pop().capitalize()
        elif len(detected) > 1:
            raise ValueError(f"Multiple sentiments detected in the text: {', '.join(detected)}\n{text}")
        else:
            raise ValueError(f"No recognizable sentiment found in the text:\n{text}")