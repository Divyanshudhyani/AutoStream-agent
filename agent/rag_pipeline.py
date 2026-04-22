import json
import os
from pathlib import Path


class AutoStreamRAG:

    def __init__(self, kb_path: str = None):
        if kb_path is None:
            base_dir = Path(__file__).parent.parent
            kb_path = base_dir / "knowledge_base" / "autostream_kb.json"
        
        with open(kb_path, "r") as f:
            self.knowledge_base = json.load(f)
        
        self._build_chunks()

    def _build_chunks(self):
        self.chunks = []
        kb = self.knowledge_base

        # Company overview
        company = kb["company"]
        self.chunks.append({
            "topic": "company_overview",
            "keywords": ["autostream", "company", "what", "about", "product"],
            "content": (
                f"AutoStream is a SaaS platform: {company['description']} "
                f"Tagline: {company['tagline']}"
            )
        })

        # Pricing plans
        for plan in kb["plans"]:
            features_str = ", ".join(plan["features"])
            self.chunks.append({
                "topic": f"plan_{plan['name'].lower().replace(' ', '_')}",
                "keywords": ["price", "pricing", "plan", "cost", "feature", "basic", "pro",
                             plan["name"].lower(), str(plan["price_monthly"])],
                "content": (
                    f"{plan['name']}: ${plan['price_monthly']}/month. "
                    f"Features: {features_str}. "
                    f"Best for: {plan['best_for']}."
                )
            })

        # All plans together (for general pricing queries)
        plans_summary = " | ".join([
            f"{p['name']} ${p['price_monthly']}/mo" for p in kb["plans"]
        ])
        self.chunks.append({
            "topic": "pricing_summary",
            "keywords": ["price", "pricing", "plan", "cost", "how much", "plans", "options"],
            "content": f"AutoStream pricing overview: {plans_summary}. "
                       f"Basic is for starters, Pro is for professionals."
        })

        # Policies
        for policy in kb["policies"]:
            self.chunks.append({
                "topic": f"policy_{policy['topic'].lower().replace(' ', '_')}",
                "keywords": [
                    policy["topic"].lower(), "policy", "refund", "support", "cancel",
                    "trial", "free"
                ],
                "content": f"{policy['topic']}: {policy['details']}"
            })

        # FAQs
        for faq in kb["faqs"]:
            self.chunks.append({
                "topic": "faq",
                "keywords": faq["question"].lower().split(),
                "content": f"Q: {faq['question']} A: {faq['answer']}"
            })

    def retrieve(self, query: str, top_k: int = 4) -> str:
        """
        Retrieve the most relevant knowledge base chunks for a given query.
        
        Args:
            query: User's question or message
            top_k: Number of top chunks to return
        
        Returns:
            Concatenated string of relevant context
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored_chunks = []
        for chunk in self.chunks:
            score = 0
            for keyword in chunk["keywords"]:
                if keyword in query_lower:
                    score += 2
                # Partial word match bonus
                for word in query_words:
                    if keyword in word or word in keyword:
                        score += 1
            scored_chunks.append((score, chunk))

        # Sort by score, descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        # If nothing matched, return full pricing + policy context
        if scored_chunks[0][0] == 0:
            relevant = [c["content"] for c in self.chunks if "plan" in c["topic"] or "policy" in c["topic"]]
        else:
            relevant = [c["content"] for score, c in scored_chunks[:top_k] if score > 0]

        return "\n\n".join(relevant) if relevant else "No specific information found."

    def get_full_context(self) -> str:
        return "\n\n".join([chunk["content"] for chunk in self.chunks])
