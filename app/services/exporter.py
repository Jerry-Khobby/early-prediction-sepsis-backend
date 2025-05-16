from fpdf import FPDF
import smtplib
from email.message import EmailMessage

def generate_pdf(report_text, output_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, report_text)
    pdf.output(output_path)

def send_email(to_email, subject, body, attachment_path):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = "noreply@yourdomain.com"
    msg["To"] = to_email
    msg.set_content(body)

    with open(attachment_path, "rb") as f:
        file_data = f.read()
        file_name = f.name
    msg.add_attachment(file_data, maintype="application", subtype="pdf", filename=file_name)

    with smtplib.SMTP("smtp.yourdomain.com", 587) as server:
        server.starttls()
        server.login("username", "password")
        server.send_message(msg)
