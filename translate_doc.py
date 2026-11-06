from docx import Document
from docx.oxml.ns import qn
p='C:/Users/objn/Desktop/sampleagent/outputs/'
d=Document(p+'Support_Agent_Brief-1.docx')
translations=[
'Support Ticket Agent System',
'The agent triages support requests, gathers evidence, and decides whether to respond directly, route the request to a specialist team, or escalate it for human review by opening a chat or sending an email.',
'Node Review','Name','Responsibility',
'record_user_message','Record the message and assign the user and session.',
'guardrail','Protect against prompt injection and NSFW content.',
'classify','Assess urgency and the need for human review.',
'extract','Extract the product, issue category, sentiment, and a summary of the issue.',
'plan','Select tools as needed and define the scope of the task.',
'knowledge_base','Retrieve relevant documents.',
'customer_history','Retrieve the user’s conversation history.',
'assess_history_relevance','Handle topic changes within the same session by returning only conversation history relevant to the user’s latest question.',
'decide','Choose an action based on risk, requiring human review for Critical cases and a specialist team response for High cases.',
'auto_respond','Draft an evidence-based response so the agent can provide information or assistance.',
'route_specialist','Open a chat session with the responsible team (MOCKUP).',
'escalate_human','Forward the case by email for human review (MOCKUP).',
'record_agent_response','Summarize the actions taken and respond to the user.',
'Architecture','Technology','Rationale',
'Python and LangGraph','StateGraph and conditional edges divide the workflow into nodes, making execution order and routing explicit and easy to test.',
'LangChain and ChatOpenAI','Separate prompts by responsibility and support convenient tool calls, simplifying development.',
'PostgreSQL and pgvector','Store chat history and support RAG for the knowledge base.',
'FastAPI and RAG worker','Provide APIs for file uploads, search, and status checks. Offload embedding generation to a worker to reduce HTTP request waiting time.',
'Docker Compose and Redis','Simplify deployment and provide tool-caching code.',
'What could go wrong and how would you handle it?',
'Add a guardrail to detect prompt injection and requests for sensitive or inappropriate content. If a request is flagged, respond with a refusal and an explanation.',
'Add assess_history_relevance to handle sudden topic changes or users continuing a conversation without closing the previous session.',
'How would you evaluate this agent in production?',
'Average cost per case, such as token costs.',
'Error rate and the rate of agent stalls or missing responses.',
'Recall of urgency classification: how reliably the agent identifies cases at each urgency level.',
'RAG retrieval recall and whether responses are supported by evidence.',
'Prompt injection attack success rate and the rate of incorrectly refusing legitimate questions.',
'Actual resolution rate, repeat contact rate, and CSAT. Have humans review sampled traces, rerun tests before changing the model or prompts, and roll out changes gradually with defined stop or rollback criteria.'
]
paras=[p for p in d.element.body.iter(qn('w:p')) if ''.join(p.itertext()) and p.findall('.//'+qn('w:t'))]
# Select only paragraphs containing visible text.
paras=[p for p in paras if ''.join(t.text or '' for t in p.findall('.//'+qn('w:t'))).strip()]
assert len(paras)==len(translations),(len(paras),len(translations))
for p,s in zip(paras,translations):
 ts=p.findall('.//'+qn('w:t')); ts[0].text=s
 for t in ts[1:]: t.text=''
d.save(pth:=p if False else 'C:/Users/objn/Desktop/sampleagent/outputs/Support_Agent_Brief_English.docx')
print(pth)
