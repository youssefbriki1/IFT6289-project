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
