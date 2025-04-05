from langchain_core.output_parsers import BaseOutputParser
import re

class ParaphraseOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        # Remove <think> sections if present
        if "<think>" in text:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        # Try to extract quoted text
        quoted = re.findall(r'"(.*?)"', text)
        if quoted:
            # Return last quoted string, stripped of whitespace
            return quoted[-1].strip()

        # Fallback: try the last line
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        if lines and len(lines[-1].split()) >= 3:
            return lines[-1].strip().strip('"')  # Remove accidental surrounding quotes

        raise ValueError(f"Failed to parse a paraphrased sentence from:\n{text}")
