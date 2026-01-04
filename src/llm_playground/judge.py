from abc import ABC, abstractmethod
import os
from openai import OpenAI

class Judge(ABC):
    """
    Abstract base class for a grading judge.
    """
    
    @abstractmethod
    def grade(self, question: str, reference: str, answer: str) -> int:
        """
        Grades the student answer against the reference.
        Returns an integer score (1-5).
        """
        pass

    def __call__(self, question: str, reference: str, answer: str) -> int:
        """
        Makes the instance callable so it works with evaluate.grade_results.
        """
        return self.grade(question, reference, answer)

class OpenAIJudge(Judge):
    def __init__(self, model_name: str = "gpt-4.1-mini", api_key: str | None = None):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def grade(self, question: str, reference: str, answer: str) -> int:
        prompt = f"""
        You are a grading assistant. 
        Question: {question}
        Reference Answer: {reference}
        Student Answer: {answer}
        
        Rate the Student Answer on a scale of 1-5 based on accuracy compared to the Reference Answer. 
        Return ONLY the integer number.
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            content = completion.choices[0].message.content.strip()
            return int(content)
        except Exception as e:
            print(f"Judge error: {e}")
            return 1
