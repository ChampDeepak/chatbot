import re
class ChunkData:
    def chunk_enterprise_doc(self, text: str, document_name: str):
        chunks = []
        lines = text.split("\n")

        title = None
        current_h2 = None
        current_h3 = None
        current_content = []

        def save_chunk():
            if current_content:
                heading = current_h3 if current_h3 else current_h2
                chunks.append({
                    "text": f"{heading}\n" + "\n".join(current_content).strip(),
                    "metadata": {
                        "document_name": document_name,
                        "title": title,
                        "heading": heading
                    }
                })
    
        for line in lines:
            h1_match = re.match(r'^# (.+)', line)
            h2_match = re.match(r'^## (.+)', line)
            h3_match = re.match(r'^### (.+)', line)
    
            if h1_match:
                title = h1_match.group(1).strip()
                continue
            
            if h2_match:
                save_chunk()
                current_h2 = h2_match.group(1).strip()
                current_h3 = None
                current_content = []
                continue
            
            if h3_match:
                save_chunk()
                current_h3 = h3_match.group(1).strip()
                current_content = []
                continue
            
            if line.strip():
                current_content.append(line)
    
        # Save last chunk
        save_chunk()

        return chunks