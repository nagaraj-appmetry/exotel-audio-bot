class Config:
    SALES_REP_NAME = "Sarah"
    SALES_BOT_NAME = "Sarah"
    COMPANY_NAME = "Appmetry Technologies"
    PRODUCTS = [
        {
            "name": "CRM Software",
            "price": "$99/month",
            "description": "Complete customer relationship management"
        }
    ]
    LANGUAGE_CODE = "en-US" 
    TEMPERATURE = 0.7
    SALES_RESPONSES = {
        "greeting": "Hi! I'm {bot_name} from {company}. How can I help?",
        "product_overview": "We offer {products}. What interests you?",
    }
    LEAD_SCORE_FACTORS = {
        "company_size": {"small": 1, "medium": 3, "large": 5},
        "budget": {"low": 1, "medium": 3, "high": 5},
        "timeline": {"immediate": 5, "soon": 3, "later": 1},
        "interest_level": {"high": 5, "medium": 3, "low": 1}
    }
    STT_ENGINE = "google"
    SAMPLE_RATE = 8000
    CRM_WEBHOOK_URL = "https://your-crm.com/webhook/leads"
    SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/your/webhook"
    BUFFER_SIZE_MS = 200
    OPENAI_MODEL = "gpt-4o-realtime"
    SERVER_HOST = "0.0.0.0"
    SERVER_PORT = 5000
    LEAD_SCORE_THRESHOLD = 12
    SESSION_TIMEOUT_MINUTES = 30
    MAX_FAILED_LOGINS = 5
    BUFFER_SIZE_MS = 5000           # Smaller audio chunk size for lower buffering delay
    SILENCE_DURATION_MS = 1000     # Shorter silence wait time before bot responds
    NOISE_THRESHOLD = 300          # Less aggressive noise filtering (clear environment)
    VAD_THRESHOLD = 0.3    
    PREFIX_PADDING_MS = 100
    LOCAL_BOT_URL = "http://127.0.0.1:8000/reply"   # your local bot server endpoint (POST)
    TTS_ENGINE = "gtts"  # or 'pyttsx3' or 'elevenlabs' (we implement the gTTS path here)
    TTS_CHUNK_MS = 200   # chunk duration when streaming back to Exotel


    @classmethod
    def validate(cls):
        pass   

    @classmethod
    def get_sales_instructions(cls):
        product_names = ', '.join([product['name'] for product in cls.PRODUCTS])
        return f"""
        You are {cls.SALES_REP_NAME}, a helpful sales assistant for {cls.COMPANY_NAME}.
        Your role is to engage customers with friendly, professional conversation,
        provide product information and pricing, qualify leads based on their needs,
        and escalate to a human sales rep if necessary.
        Products offered: {product_names}
        """

