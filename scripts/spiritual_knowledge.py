import json
import os
from typing import Dict, List, Optional

class SpiritualKnowledge:
    def __init__(self):
        self.knowledge_base = {
            "biblical_concepts": {
                "love": {
                    "definition": "Agape love - unconditional, sacrificial love as demonstrated by God",
                    "key_verses": [
                        "1 Corinthians 13:4-7",
                        "John 3:16",
                        "1 John 4:7-8"
                    ],
                    "themes": ["sacrifice", "unconditional", "forgiveness"]
                },
                "faith": {
                    "definition": "Confidence in what we hope for and assurance about what we do not see",
                    "key_verses": [
                        "Hebrews 11:1",
                        "Matthew 17:20",
                        "James 2:17"
                    ],
                    "themes": ["trust", "belief", "action"]
                },
                "grace": {
                    "definition": "God's unmerited favor and love towards humanity",
                    "key_verses": [
                        "Ephesians 2:8-9",
                        "Romans 5:8",
                        "2 Corinthians 12:9"
                    ],
                    "themes": ["forgiveness", "mercy", "unmerited favor"]
                }
            },
            "theological_frameworks": {
                "salvation": {
                    "concept": "God's plan for reconciling humanity to Himself",
                    "key_aspects": [
                        "God's love",
                        "Human sin",
                        "Jesus' sacrifice",
                        "Faith response"
                    ]
                },
                "sanctification": {
                    "concept": "The process of becoming more like Christ",
                    "key_aspects": [
                        "Holy Spirit's work",
                        "Spiritual growth",
                        "Obedience",
                        "Transformation"
                    ]
                }
            },
            "spiritual_practices": {
                "prayer": {
                    "types": [
                        "Adoration",
                        "Confession",
                        "Thanksgiving",
                        "Supplication"
                    ],
                    "guidelines": [
                        "Be honest",
                        "Be persistent",
                        "Be thankful",
                        "Be still"
                    ]
                },
                "meditation": {
                    "focus": [
                        "God's Word",
                        "His character",
                        "His promises",
                        "His presence"
                    ],
                    "benefits": [
                        "Spiritual growth",
                        "Peace",
                        "Wisdom",
                        "Strength"
                    ]
                }
            }
        }
        
    def get_concept_info(self, concept: str) -> Optional[Dict]:
        """Get information about a specific spiritual concept."""
        # Check biblical concepts
        if concept in self.knowledge_base["biblical_concepts"]:
            return self.knowledge_base["biblical_concepts"][concept]
            
        # Check theological frameworks
        if concept in self.knowledge_base["theological_frameworks"]:
            return self.knowledge_base["theological_frameworks"][concept]
            
        # Check spiritual practices
        if concept in self.knowledge_base["spiritual_practices"]:
            return self.knowledge_base["spiritual_practices"][concept]
            
        return None
        
    def get_related_concepts(self, concept: str) -> List[str]:
        """Get related spiritual concepts."""
        related = []
        
        # Check biblical concepts
        if concept in self.knowledge_base["biblical_concepts"]:
            info = self.knowledge_base["biblical_concepts"][concept]
            related.extend(info.get("themes", []))
            
        # Check theological frameworks
        if concept in self.knowledge_base["theological_frameworks"]:
            info = self.knowledge_base["theological_frameworks"][concept]
            related.extend(info.get("key_aspects", []))
            
        # Check spiritual practices
        if concept in self.knowledge_base["spiritual_practices"]:
            info = self.knowledge_base["spiritual_practices"][concept]
            related.extend(info.get("types", []) + info.get("guidelines", []))
            
        return related
        
    def format_concept_prompt(self, concept: str) -> str:
        """Format a prompt for the model about a spiritual concept."""
        info = self.get_concept_info(concept)
        if not info:
            return ""
            
        prompt = f"Regarding the spiritual concept of {concept}:\n\n"
        
        if "definition" in info:
            prompt += f"Definition: {info['definition']}\n\n"
            
        if "key_verses" in info:
            prompt += "Key Bible verses:\n"
            for verse in info["key_verses"]:
                prompt += f"- {verse}\n"
            prompt += "\n"
            
        if "themes" in info:
            prompt += "Related themes:\n"
            for theme in info["themes"]:
                prompt += f"- {theme}\n"
            prompt += "\n"
            
        if "key_aspects" in info:
            prompt += "Key aspects:\n"
            for aspect in info["key_aspects"]:
                prompt += f"- {aspect}\n"
            prompt += "\n"
            
        return prompt

if __name__ == "__main__":
    # Test the spiritual knowledge base
    knowledge = SpiritualKnowledge()
    
    # Test concept lookup
    concept = "love"
    info = knowledge.get_concept_info(concept)
    print(f"Information about {concept}:")
    print(json.dumps(info, indent=2))
    
    # Test related concepts
    related = knowledge.get_related_concepts(concept)
    print(f"\nRelated concepts for {concept}:")
    print(related)
    
    # Test prompt formatting
    prompt = knowledge.format_concept_prompt(concept)
    print(f"\nFormatted prompt for {concept}:")
    print(prompt) 