from dataclasses import field
import logging
from typing import Optional, Dict, List, Any, Union, Generator

from .base_tool import BaseTool 
from ..models.embedders.base import BaseEmbedder  
from ..models.chat_models.base import BaseChatModel
from ..utils.prompts import ERD_DIAGRAM_GENERATION_PROMPT


class ERDDiagramGenerator(BaseTool):
    name: str = "erd_diagram_generator"
    description: str = """Given a natural language description of a data model, this tool converts it into a valid Mermaid `erDiagram` code block. Use this tool when the user asks for an Entity Relationship Diagram (ERD)."""
    register: bool = False
    stream: bool = True

    def __init__(
        self,
        llm: BaseChatModel,
        embedder: BaseEmbedder,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(llm, embedder, logger)
        self.prompt = ERD_DIAGRAM_GENERATION_PROMPT
        self.top_k: int = 10
        self.special_token: str = " [__ERD__] "

    def __call__(
        self,
        user_query: str,
        chat_history: List[dict] = [],
        **kwargs: Any,
    ) -> Optional[Union[dict, Generator[str, None, None]]]:
        """Generate ERD diagram code using the LLM."""
        session_id = kwargs.get("session_id", None)
        vectorstore = kwargs.get("vectorstore", None)
        project_tree = kwargs.get("project_tree", None)
        api_key = kwargs.get("api_key", None)
        stream = kwargs.get("stream", self.stream)
        if project_tree:            
            system_prompt, user_prompt, max_new_tokens = self.get_architecture_context(
                user_query=user_query,
                prompt=self.prompt,
                session_id=session_id,
                project_tree=project_tree,
                vectorstore=vectorstore,
                chat_history=chat_history,
                top_k=self.top_k,
            )
        else:
            system_prompt, user_prompt, max_new_tokens = self.build_db_context(
                session_id=session_id,
                user_query=user_query,
                prompt=self.prompt,
                vectorstore=vectorstore,
                top_k=self.top_k,
                chat_history=chat_history,
            )
        mermaid_code = ""
        response = self.llm_call(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            chat_history=chat_history,
            temperature=self.temperature,
            max_new_tokens=max_new_tokens,
            stream=stream
        )
        mermaid_code = ''
        if stream:
            for chunk in response:
                mermaid_code += chunk['choices'][0]['message']['content']
                yield chunk
        else:
            mermaid_code = response['choices'][0]['message']['content']
            yield response

        mermaid_code = mermaid_code.replace("```", "").replace("mermaid", "").strip()
        # Optional: Send to Mermaid renderer and return base64 image
        # TODO validator mermaid_code ---
        # output_path = os.path.join(self.tmp_dir, f"{str(uuid.uuid1())}.png")
        # if self.__mermaid_save_diagram(
        #     mermaid_code=mermaid_code,
        #     output_path=output_path,
        #     image_format = "png",
        #     background_color="white",
        #     scale=2.0,
        # ):
        #     mime_type, _ = mimetypes.guess_type(output_path)
        #     base64_str = f''
        #     with open(output_path, "rb") as image_file:
        #         base64_str = base64.b64encode(image_file.read()).decode('utf-8')
        #     base64_with_metadata = f"\n\ndata:{mime_type};base64,{base64_str}"
            
        #     # TODO: wrape this in chat api
        #     yield chat_completion_wrapper(llm.call_id, llm.model_name, base64_with_metadata)
        #     if os.path.exists(output_path):
        #         os.remove(output_path)

    def __str__(self):
        return f"{self.name}: {self.description}"