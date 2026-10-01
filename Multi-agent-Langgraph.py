import os
import re
import smtplib
import time
from email.message import EmailMessage
from typing import TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# PDF libraries
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


# ============================================================
# 1. GROQ API KEY
# ============================================================

api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY environment variable is not set."
    )


# ============================================================
# 2. INITIALIZE GROQ LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.3,
    api_key=api_key
)


# ============================================================
# 3. SHARED STATE
# ============================================================

class AgentState(TypedDict):
    question: str
    research: str
    technical: str
    report: str


# ============================================================
# 4. LLM INVOKE WITH RATE-LIMIT RETRY
# ============================================================

def invoke_llm(messages, max_tokens=1200, retries=3):

    for attempt in range(retries):

        try:

            response = llm.invoke(
                messages,
                max_tokens=max_tokens
            )

            return response

        except Exception as e:

            error_message = str(e)

            # Handle Groq 429 rate limit
            if "429" in error_message or "RateLimitError" in type(e).__name__:

                if attempt < retries - 1:

                    wait_time = 2 ** attempt

                    print(
                        f"\n[Rate Limit] Groq rate limit reached."
                    )

                    print(
                        f"[Rate Limit] Waiting {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                else:

                    print(
                        "\n[Error] Groq rate limit still active."
                    )

                    raise

            else:

                raise


# ============================================================
# 5. COORDINATOR AGENT
# ============================================================

def coordinator(state: AgentState):

    print("\n[Coordinator] Starting workflow...")

    return {
        "research": "",
        "technical": "",
        "report": ""
    }


# ============================================================
# 6. RESEARCH AGENT
# ============================================================

def research_agent(state: AgentState):

    print("\n[Research Agent] Working...")

    prompt = f"""
You are an IT Research Agent.

Analyze the following question.

Identify:

Important concepts
Requirements
Benefits
Challenges

Keep your answer concise and focused.
Do not repeat information.

Question:
{state['question']}
"""

    response = invoke_llm(
        [
            SystemMessage(
                content="You are a concise IT Research Agent."
            ),
            HumanMessage(
                content=prompt
            )
        ],
        max_tokens=1000
    )

    return {
        "research": response.content
    }


# ============================================================
# 7. TECHNICAL AGENT
# ============================================================

def technical_agent(state: AgentState):

    print("\n[Technical Agent] Working...")

    prompt = f"""
You are a Kubernetes Technical Architect.

Based on the research below, propose a concise
technical solution.

Include:

Architecture
Kubernetes components
Deployment approach
Security
Monitoring

Avoid unnecessary explanations and repetition.

User question:
{state['question']}

Research findings:
{state['research']}
"""

    response = invoke_llm(
        [
            SystemMessage(
                content="You are a concise Kubernetes Technical Architect."
            ),
            HumanMessage(
                content=prompt
            )
        ],
        max_tokens=1200
    )

    return {
        "technical": response.content
    }


# ============================================================
# 8. REPORT AGENT
# ============================================================

def report_agent(state: AgentState):

    print("\n[Report Agent] Working...")

    prompt = f"""
You are a Technical Report Agent.

Create a concise professional technical report.

Use exactly these sections:

1. Executive Summary
2. Research Findings
3. Technical Architecture
4. Implementation Steps
5. Security
6. Monitoring
7. Conclusion

Keep the report concise.

Do not repeat information.

Do not invent unsupported facts.

User question:
{state['question']}

Research findings:
{state['research']}

Technical solution:
{state['technical']}
"""

    response = invoke_llm(
        [
            SystemMessage(
                content="You are a concise Technical Report Agent."
            ),
            HumanMessage(
                content=prompt
            )
        ],
        max_tokens=1200
    )

    return {
        "report": response.content
    }


# ============================================================
# 9. ESCAPE TEXT FOR PDF
# ============================================================

def escape_text(text):

    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# ============================================================
# 10. SAVE REPORT AS PDF
# ============================================================

def save_report_as_pdf(
    report,
    question,
    filename="technical_report.pdf"
):

    print("\n[PDF Generator] Creating PDF...")

    # Create PDF document
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    # Get default styles
    styles = getSampleStyleSheet()

    # --------------------------------------------------------
    # Title style
    # --------------------------------------------------------

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=26,
        alignment=TA_CENTER,
        textColor=colors.darkblue,
        spaceAfter=20
    )

    # --------------------------------------------------------
    # Question style
    # --------------------------------------------------------

    question_style = ParagraphStyle(
        "Question",
        parent=styles["BodyText"],
        fontSize=11,
        leading=16,
        textColor=colors.black,
        spaceAfter=15
    )

    # --------------------------------------------------------
    # Heading style
    # --------------------------------------------------------

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.darkblue,
        spaceBefore=12,
        spaceAfter=8
    )

    # --------------------------------------------------------
    # Body style
    # --------------------------------------------------------

    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10,
        leading=15,
        spaceAfter=7
    )

    # --------------------------------------------------------
    # Bullet style
    # --------------------------------------------------------

    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-8,
        spaceAfter=5
    )

    # --------------------------------------------------------
    # Create PDF content
    # --------------------------------------------------------

    story = []

    # Title
    story.append(
        Paragraph(
            "Technical Architecture Report",
            title_style
        )
    )

    # User question
    story.append(
        Paragraph(
            f"<b>User Question:</b> "
            f"{escape_text(question)}",
            question_style
        )
    )

    story.append(
        Spacer(1, 10)
    )

    # --------------------------------------------------------
    # Process report line by line
    # --------------------------------------------------------

    lines = report.split("\n")

    for line in lines:

        line = line.strip()

        # Empty line
        if not line:

            story.append(
                Spacer(1, 6)
            )

            continue

        # Remove Markdown bold
        line = re.sub(
            r"\*(.*?)\*",
            r"\1",
            line
        )

        # Remove Markdown italic
        line = re.sub(
            r"\_(.*?)\_",
            r"\1",
            line
        )

        # ----------------------------------------------------
        # Markdown headings
        # ----------------------------------------------------

        if line.startswith("#"):

            heading = line.lstrip("#").strip()

            story.append(
                Paragraph(
                    escape_text(heading),
                    heading_style
                )
            )

        # ----------------------------------------------------
        # Numbered sections
        # ----------------------------------------------------

        elif re.match(
            r"^\d+\.\s+",
            line
        ):

            story.append(
                Paragraph(
                    escape_text(line),
                    heading_style
                )
            )

        # ----------------------------------------------------
        # Bullet points
        # ----------------------------------------------------

        elif line.startswith("- ") or line.startswith("* "):

            bullet = line[2:].strip()

            story.append(
                Paragraph(
                    f"• {escape_text(bullet)}",
                    bullet_style
                )
            )

        # ----------------------------------------------------
        # Normal paragraph
        # ----------------------------------------------------

        else:

            story.append(
                Paragraph(
                    escape_text(line),
                    body_style
                )
            )

    # --------------------------------------------------------
    # Generate PDF
    # --------------------------------------------------------

    doc.build(story)

    print(
        "\n[PDF Generator] PDF created successfully!"
    )

    print(
        f"[PDF Generator] File: {filename}"
    )


# ============================================================
# 11. SEND EMAIL WITH PDF ATTACHMENT
# ============================================================

def send_email_with_pdf(
    recipient_email: str,
    pdf_filename: str = "technical_report.pdf",
    subject: str = "Technical Architecture Report"
):

    print(f"\n[Email Exporter] Sending email to {recipient_email}...")

    sender_email = os.environ.get("SENDER_EMAIL")
    sender_password = os.environ.get("SENDER_PASSWORD")

    if not sender_email or not sender_password:

        print(
            "[Email Error] SENDER_EMAIL or SENDER_PASSWORD environment variables are not set."
        )

        return

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg.set_content(
        "Hello,\n\nPlease find attached the generated Technical Architecture Report.\n\nBest regards,\nLangGraph Multi-Agent System"
    )

    # Read and attach PDF file
    try:

        with open(pdf_filename, 'rb') as f:

            file_data = f.read()

            msg.add_attachment(
                file_data,
                maintype='application',
                subtype='pdf',
                filename=pdf_filename
            )

    except FileNotFoundError:

        print(
            f"[Email Error] Attachment file '{pdf_filename}' not found."
        )

        return

    # Send using standard SMTP (Gmail Port 587 + STARTTLS)
    try:

        with smtplib.SMTP('smtp.gmail.com', 587) as server:

            server.starttls()

            server.login(sender_email, sender_password)

            server.send_message(msg)

        print(
            f"[Email Exporter] Email successfully sent to {recipient_email}!"
        )

    except Exception as e:

        print(
            f"[Email Error] Failed to send email: {e}"
        )


# ============================================================
# 12. BUILD LANGGRAPH
# ============================================================

graph = StateGraph(AgentState)


# Add nodes
graph.add_node(
    "coordinator",
    coordinator
)

graph.add_node(
    "research",
    research_agent
)

graph.add_node(
    "technical",
    technical_agent
)

graph.add_node(
    "report",
    report_agent
)


# ============================================================
# 13. DEFINE WORKFLOW
# ============================================================

graph.add_edge(
    START,
    "coordinator"
)

graph.add_edge(
    "coordinator",
    "research"
)

graph.add_edge(
    "research",
    "technical"
)

graph.add_edge(
    "technical",
    "report"
)

graph.add_edge(
    "report",
    END
)


# ============================================================
# 14. COMPILE GRAPH
# ============================================================

app = graph.compile()


# ============================================================
# 15. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("       LANGGRAPH MULTI-AGENT SYSTEM")
    print("=" * 60)

    # Get user question
    question = input(
        "\nEnter your question: "
    )

    # --------------------------------------------------------
    # Run LangGraph
    # --------------------------------------------------------

    try:

        result = app.invoke({

            "question": question,

            "research": "",

            "technical": "",

            "report": ""
        })

    except Exception as e:

        print("\n========================================")
        print("ERROR")
        print("========================================")

        print(e)

        print(
            "\nIf the error is a Groq 429 rate-limit error,"
        )

        print(
            "wait a few seconds and run the program again."
        )

        raise


    # --------------------------------------------------------
    # Display final report
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("                 FINAL REPORT")
    print("=" * 60)

    print(
        result["report"]
    )


    # --------------------------------------------------------
    # Save PDF
    # --------------------------------------------------------

    pdf_filename = "technical_report.pdf"

    save_report_as_pdf(
        report=result["report"],
        question=question,
        filename=pdf_filename
    )


    # --------------------------------------------------------
    # Send Email
    # --------------------------------------------------------

    recipient = input(
        "\nEnter recipient email to send report (or press Enter to skip): "
    ).strip()

    if recipient:

        send_email_with_pdf(
            recipient_email=recipient,
            pdf_filename=pdf_filename
        )


    # --------------------------------------------------------
    # Completed
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("WORKFLOW COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(
        f"\nPDF file: {pdf_filename}"
    )
