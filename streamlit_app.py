import streamlit as st
from gliner import GLiNER
from sentence_transformers import SentenceTransformer
import torch
import re

st.set_page_config(page_title="Twitter Bio Analyzer", layout="centered")

# --- Load model once ---
@st.cache_resource
def load_models():
    gliner_model = GLiNER.from_pretrained("EmergentMethods/gliner_medium_news-v2.1")
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return gliner_model, embedding_model

gliner_model, embedding_model = load_models()

occupations_dict = {
    'accounting': ['accountant', 'auditor', 'ca', 'chartist', 'cpa', 'enrolledagent', 'gstpractioner', 'practicingca'],
    'advertising': ['advertisingmanager', 'copywriter', 'campusambassador', 'commsstrategist', 'communicator', 'corporatecommunications', 'corporatecommunicator', 'evangelist', 'influencermarketing', 'marketingandsales', 'marketingofficer', 'marketresearch', 'digitalcommunicationsandcontentmarketingspecialist', 'digitalmarketer', 'digitalmarketing', 'digitalmarketingprofesional', 'digitalmarketingstrategist'],
    'aerospace': ['aerospaceengineer', 'flighttestengineer', 'missioncontroller'],
    'agriculture': ['farmer', 'dairycattleproduction', 'deepagriculture', 'farming', 'horticulturist'],
    'animation': ['animator', 'storyboardartist', '3dmodeler'],
    'anthropology': ['anthropologist', 'archaeologist', 'culturalresourcemanager'],
    'aquaculture': ['fishfarmer', 'aquaculturist', 'marinetechnician'],
    'archeometry': ['archaeologicalscientist', 'artifactanalyst'],
    'archival_science': ['archivist', 'recordsmanager'],
    'architecture': ['architect', 'interiordesigner', 'commercialinteriordesigner'],
    'art': ['artist', 'artdirector', 'ceramicartist', 'kilntechnician', 'glazetechnologist', 'leathercraftartist', 'saddlemaker', 'lacemaker', 'textiledesigner', 'calligrapher', 'handletteringartist', 'kalamkar'],
    'astronomy': ['astronomer', 'astrophysicist', 'planetariumdirector'],
    'audiology': ['audiologist', 'hearingaidtechnician'],
    'automotive': ['cardesigner', 'mechanic'],
    'aviation': ['airtrafficcontroller', 'flightattendant', 'aviationmechanic', 'pilot'],
    'banking': ['bankmanager', 'loanofficer', 'exbanker'],
    'beekeeping': ['beekeeper', 'apiarymanager'],
    'biotechnology': ['biotechnologist', 'researchscientist'],
    'blockchain': ['blockchaindeveloper', 'cryptoeconomist', 'smartcontractengineer', 'crypto trader', 'crypto educator', 'crypto investor', 'crypto enthusiast', 'crypto researcher', 'web3 enthusiast', 'web3 founder', 'tokenomics advisor', 'nft artist', 'nft collector', 'nft trader', 'nft flipper', 'meme coin investor', 'defi researcher', 'web3 developer'],
    'broadcasting': ['radioproducer', 'tvnewsanchor'],
    'business': ['businessanalyst', 'entrepreneur', 'ecommerce', 'enterpreneur', 'entrepreneurship', 'fractionalgtmstrategist', 'freelancer', 'co-founder', 'cofounder', 'owner', 'partner', 'lijwaniyatraders'],
    'calligraphy': ['calligrapher', 'handletteringartist'],
    'cartography': ['cartographer', 'mapanalyst', 'gisanalyst'],
    'ceremonial_services': ['funeraldirector', 'mortician', 'embalmer'],
    'classic_studies': ['classicist', 'latintranslator', 'greekphilologist'],
    'climatology': ['climatologist', 'paleoclimatologist', 'climatescientist', 'weatherforecaster'],
    'color_theory': ['colortheorist', 'colorconsultant'],
    'compliance': ['complianceofficer', 'regulatoryaffairsmanager'],
    'computer_science': ['programmer', 'datascientist', 'softwaredeveloper', 'testengineer', 'c++', 'certifiedethicalhacker', 'chieftinkerer', 'co-mod', 'coder', 'coding', 'cybersecurity', 'cybersecurityexpert', 'cybersecurityenthusiast', 'cybervillager', 'cse', 'dataanalysis', 'datascience', 'deeplearning', 'dev', 'devops', 'documentationanduserassistanceprofessional', 'gadgetlover', 'gadgets', 'geek', 'indiemaker', 'informationdeveloper', 'it', 'itdecisionmakers', 'itfield', 'itman', 'itmember', 'itprofessional', 'javabackenddeveloper', 'metacertifiedfrontenddeveloper', 'formersiliconvalleydrone', 'fulltimeqa'],
    'construction': ['plumber', 'carpenter', 'electrician', 'construction', 'contractor', 'epccontractor', 'greenbuilding'],
    'consulting': ['consultant', 'consulting', 'crmmanagementconsultant', 'exkpmg', 'formercountryconsultant', 'industryexperts', 'internationalconsultant'],
    'consumer_research': ['consumerinsightsanalyst', 'usabilityresearcher'],
    'criminal_justice': ['criminalinvestigator', 'forensicscientist', 'police'],
    'cryogenics': ['cryogenicsengineer', 'refrigerationspecialist'],
    'customer_service': ['customerservicerepresentative', 'supportspecialist', 'helpdeskagent'],
    'customs_immigration': ['customsofficer', 'immigrationofficer'],
    'cybersecurity': ['securityanalyst', 'ethicalhacker', 'cybersecurityengineer'],
    'data_science': ['datascientist', 'dataanalyst', 'machinelearningengineer'],
    'defense_military': ['militaryofficerintelligenceanalyst', 'exairforce', 'exnavy', 'jawan', 'defence'],
    'dentistry': ['dentist', 'orthodontist', 'periodontist'],
    'diplomacy': ['diplomat', 'foreignserviceofficer', 'ambassador', 'consul'],
    'economics': ['economist', 'financialplanner'],
    'ecommerce': ['ecommercespecialist', 'onlinestoremanager', 'digitalmarketer'],
    'education': ['student', 'teacher', 'professor', 'lecturer', 'educationist', 'careercounseling', 'coaching', 'msc', 'mentor', 'principal', 'iitian', 'intellectuals'],
    'emergency_management': ['emergencymanager', 'firechief', 'publicsafetydirector'],
    'energy': ['ep', 'energymanagement', 'exdvc', 'iocian', 'lpgsales', 'lpgterritory'],
    'engineering': ['engineer', 'directorrefineries', 'directorpipelines', 'electrical', 'electronics', 'engg', 'engineeringchange', 'heavyelectricalengineering', 'heavyelectricaljobs', 'industrialautomation', 'mechengg', 'mechanical', 'chiefinstallationmanager', 'civilenginner'],
    'entertainment': ['actor', 'filmdirector', 'musician', 'guitarist', 'magician', 'gamer', 'gamergirl'],
    'environmental_science': ['environmentalscientist', 'conservationbiologist', 'sustainabilityspecialist'],
    'esports': ['esportsplayer', 'esportscoach', 'streamingmanager'],
    'event_management': ['eventplanner', 'eventorganiser', 'conferencemanagement'],
    'event_technology': ['avtechnician', 'eventtechnologyspecialist'],
    'fashion': ['fashiondesigner', 'fashionbuyer', 'stylist', 'makeup-artist', 'fashionist'],
    'fermentation_science': ['fermentationscientist', 'brewmaster', 'cheesemaker'],
    'film': ['cinematographer', 'screenwriter', 'filmcritic', 'filmdistributor'],
    'finance': [
    'financialadvisor', 'investmentbanker', 'riskanalyst', 'investor', 'chiefriskofficer',
    'crypto_trader', 'equity', 'fandcs', 'financeenthusiast', 'financialfraudinvestigation', 'golder',
    'healthxwealthstrategist', 'insuranceindustry', 'insuranceprofessional', 'intradaypositionaltrader',
    'investmentconsultant', 'investmentnerd', 'nfttraderandcollector', 'optionbuyer', 'pbsharetrader',
    'directorfinance', 'districttreasurer', 'dsa', 'mbafinance',
    'day_trader', 'forex_trader', 'stock_trader', 'stock_analyst', 'market_strategist',
    'equity_analyst', 'portfolio_manager', 'nft_flipper', 'nft_investor', 'options_trader',
    'token_shiller', 'technical_analyst', 'derivatives_trader', 'options_strategist',
    'swing_trader', 'price_action_trader', 'candlestick_analyst'],
    'fisheries': ['fisheriesscientist', 'fisheryofficer'],
    'food_service': ['chef', 'restaurantmanager', 'sommelier', 'waiter', 'waitress', 'bartender', 'cook', 'cooking', 'cookingismypassion'],
    'forestry': ['forester', 'parkranger', 'timberbuyer'],
    'game_development': ['gamedesigner', 'gameprogrammer', 'leveldesigner', 'gamescripter'],
    'genealogy': ['genealogist', 'familyhistorian'],
    'gemology': ['gemologist', 'diamondgrader', 'jewelryappraiser'],
    'geology': ['geologist', 'geophysicist'],
    'government': ['civilservant', 'policyanalyst', 'publicadministrator', 'cabinetminister', 'cm', 'generalsecratary', 'generalsecretary', 'governmentofficebearer', 'governmentservant', 'ias', 'internationalrelations', 'ips', 'ir', 'jointsecretary', 'ministerofpetroleumandnaturalgas', 'mla', 'exminister', 'exmla', 'exofficer', 'formerassemblyspeaker', 'formercabinetminister', 'formermayor', 'formerpresident', 'formerstatesecretary', 'formercivilservant', 'panchayatassistant', 'policymakers', 'publicadministration', 'publicinterest', 'publicservent'],
    'graphic_design': ['graphicdesigner', 'illustrator', 'webdesigner', 'graphicsdesigner', 'design', 'designer'],
    'healthcare': ['doctor', 'nurse', 'pharmacist', 'surgeon', 'healer', 'healthleader', 'healthcare', 'intensivist', 'interventionalpulmophilanthropist', 'koronawarrior', 'mbbs', 'md', 'medico', 'mentalhealthadvisor', 'naturopathist', 'occupationalhealthconsultant', 'pathology', 'personalcounseling', 'petpsychatrist', 'pranichealer', 'celebritypulmonologist', 'clinician', 'criticalcaremedicine', 'dr', 'drjyotikumar', 'drmaddikerakrishnareddy', 'drpoonam', 'drprasantrout', 'drsandy', 'drsksrivastava', 'drumapathy', 'gp'],
    'heritage_conservation': ['conservator', 'heritagemanager'],
    'horticulture': ['horticulturist', 'landscapegardener', 'greenhousemanager'],
    'hospitality': ['hotelmanager', 'eventplanner', 'security', 'guard', 'firefighter', 'detective', 'housekeeper', 'cleaner', 'hospitality', 'hotelier'],
    'human_resources': ['hrmanager', 'recruiter', 'trainingspecialist', 'hrcb', 'hradministrator', 'hrprofessional', 'dgm-hr', 'mastersinhumanresources', 'placementsconsultant'],
    'information_technology': ['itmanager', 'networkadministrator'],
    'infrastructure': ['infra', 'infrastructure'],
    'insurance': ['insuranceagent'],
    'journalism': ['journalist', 'newsanchor', 'reporter', 'journalism', 'masscommunicationurdu', 'mediacoordinator', 'mediaincharge', 'mediapersonality', 'mediaprofessional', 'mediarelations', 'mediaworker', 'nationalbusinesseditor', 'newseditorwriter', 'onlineeditor'],
    'knitting_textiles': ['knitweardesigner', 'textileengineer'],
    'law': ['attorney', 'judge', 'paralegal', 'lawyer', 'prosecutor', 'advocate', 'certifiedmediator', 'constitutionallaw', 'constitutionalist', 'environmentallaw', 'labourlawsfactoriesact', 'law', 'legal', 'legaladvisortogovernor', 'litigation', 'ipexpert', 'publicnotary'],
    'library_science': ['librarian'],
    'lighting_design': ['lightingdesigner', 'theatricallighttechnician'],
    'linguistics': ['interpreter', 'languageteacher', 'translator', 'localizationspecialist', 'subtitler', 'signlanguageinterpreter', 'deafeducator'],
    'logistics': ['supplychainmanager', 'inventorycontroller', 'shippingcoordinator', 'deliveryboy', 'transportationplanner', 'logisticsmanager', 'driver'],
    'management': ['ceo', 'cto', 'operationsmanager', 'projectmanager', 'receptionist', 'clerk', 'c&md', 'chiefgeneralmanager', 'chiefregionalmanager', 'crisismanagement', 'decisionmaker', 'deputygeneralmanager', 'dir', 'directorplanningandbusinessdev', 'directorresearch', 'disastermanager', 'generalmanager', 'head', 'lead', 'leadprogrammanager', 'leader', 'leading', 'management', 'manager', 'managingdirector', 'negotiator', 'pm', 'programcoordinator', 'programdirector', 'projectmanagementconsultant', 'projectmanager', 'exgroupgeneralmanager', 'exhead', 'exindependantdirector'],
    'manufacturing': ['productionsupervisor', 'qualitycontrolinspector', 'electroplatingjobs', 'machinemaker', 'machinery', 'manufacturers', 'manufacturing', 'productionsupervisor'],
    'marine_biology': ['marinebiologist', 'oceanographer', 'aquaticecologist'],
    'marketing': ['marketingmanager', 'marketresearchanalyst', 'socialmediaspecialist', 'managersales', 'mandateseller', 'mdrtachiever'],
    'mathematics': ['mathematician', 'statistician', 'actuary'],
    'media': ['mediabuyer', 'mediaplanner', 'photographer', 'videographer', 'vlogger', 'blogger', 'youtuber', 'contentcreator', 'celebritymanager', 'commentor', 'contentwriter', 'creator', 'creatorofthings', 'critic', 'critic2', 'crosswordcomposer', 'currentaffairs', 'digitalcreator', 'digitalcreators', 'digitalinfluencer', 'editorinchief', 'exeditor', 'ghostwriter', 'hosting', 'influencer', 'podcast', 'pr', 'prmanager', 'producer', 'publicinfluencer'],
    'meteorology': ['meteorologist'],
    'mining': ['mines', 'miningandbusinessdevelopmentprofessional'],
    'music': ['composer', 'soundengineer', 'dj', 'musicproducer'],
    'nanotechnology': ['nanotechnologist', 'molecularengineer'],
    'nutrition': ['dietitian', 'nutritionist', 'foodscientist'],
    'occupational_therapy': ['occupationaltherapist', 'ergonomist'],
    'packaging': ['packagingindustry'],
    'paleontology': ['paleontologist', 'fossilpreparator'],
    'pet_care': ['dogtrainer', 'petgroomer', 'animalrescuespecialist', 'caninetrainer'],
    'philosophy': ['philosopher', 'ethicist', 'logician'],
    'physical_therapy': ['physicaltherapist', 'sportstherapist'],
    'physics': ['physicist'],
    'political_science': ['politician', 'publicpolicyadvisor', 'campaigner', 'karyakarta', 'liberal', 'liberalcentrist', 'politicalanalyst', 'politicalanalysts', 'politicalthinker', 'politics', 'politicsgeopolitics', 'politicsspecialist', 'precinctchair'],
    'prosthetics_orthotics': ['prosthetist', 'orthotist'],
    'psychology': ['psychologist', 'therapist'],
    'publishing': ['editor', 'bookdesigner', 'proofreader'],
    'quality_control': ['qcmanager'],
    'real_estate': ['realestateagent'],
    'religion': ['clergy', 'religiouseducator', 'chaplain', 'preacher'],
    'renewable_energy': ['solarpanelinstaller', 'windturbinetechnician', 'energyconsultant'],
    'research': ['geopoliticalanalyst', 'geopolitics', 'projectscientist'],
    'retail': ['retailmanager', 'petrolpump'],
    'robotics': ['roboticsengineer', 'automationtechnician', 'mechatronicsengineer'],
    'sales': ['salesrepresentative', 'accountexecutive', 'salesmanager', 'dealer', 'distributors'],
    'security': ['chowkidar'],
    'skilled_trades': ['electricianjobs', 'plumberjobs', 'laborer', 'laborers'],
    'social_work': ['socialworker', 'communityvolunteer', 'csr', 'developmentprofessional', 'humanrightactivist', 'humanrightsactivist', 'humanrightsfighter'],
    'sociology': ['sociologist', 'demographer'],
    'space_science': ['spacescientist', 'astrobiologist', 'spaceengineer'],
    'speech_language_pathology': ['speechlanguagepathologist', 'speechtherapist'],
    'sports': ['coach', 'athletictrainer', 'sportsanalyst', 'sportsperson', 'chessgrandmaster', 'cricketgeek', 'cricketplayer', 'cyclist', 'distancerunner', 'exswimmer', 'grandmaster', 'player', 'professionalchessplayer'],
    'tattoo_industry': ['tattooartist', 'tattoodesigner', 'tattooshopmanager'],
    'tourism': ['tourguide', 'travelagent', 'tourismmanager'],
    'toxicology': ['toxicologist', 'environmentaltoxicologist'],
    'toy_design': ['toydesigner', 'productdeveloper', 'toysafetyengineer'],
    'training': ['facilitator'],
    'transportation': ['pilot'],
    'ux_ui_design': ['uxdesigner', 'uidesigner', 'usabilityanalyst'],
    'veterinary_medicine': ['veterinarian', 'animalbehaviorist'],
    'web_development': ['webdeveloper', 'front-enddeveloper', 'back-enddeveloper'],
    'wellness': ['certifiedyogainstructor', 'fitnessenthusiast'],
    'winemaking': ['winemaker', 'viticulturist', 'oenologist'],
    'zoology': ['zoologist', 'wildlifebiologist', 'animalecologist'],
    'other_professionals': ['natureenthusiast', 'naturelover', 'numerologist', 'offroader', 'patriot', 'positivethinker', 'qualified', 'qureshi', 'privatejob', 'professional', 'productstrategist']
}

# --- Preprocess and embed keywords ---
def clean_keyword(term):
    term = re.sub(r'([a-z])([A-Z])', r'\1 \2', term)
    term = term.replace("_", " ")
    term = re.sub(r'\s+', ' ', term).strip()
    return term.lower()

occupation_phrases = []
occupation_meta = []
for main_cat, sublist in occupations_dict.items():
    for keyword in sublist:
        cleaned = clean_keyword(keyword)
        occupation_phrases.append(cleaned)
        occupation_meta.append((cleaned, main_cat, sublist))

occupation_embeddings = embedding_model.encode(
    occupation_phrases,
    convert_to_tensor=True,
    normalize_embeddings=True
)

# --- Semantic matching function ---
SIMILARITY_THRESHOLD = 0.5

def match_occupation_semantic(bio_text):
    bio_embedding = embedding_model.encode(bio_text, convert_to_tensor=True, normalize_embeddings=True)
    similarities = torch.nn.functional.cosine_similarity(bio_embedding, occupation_embeddings)
    top_idx = torch.argmax(similarities).item()
    top_score = similarities[top_idx].item()

    if top_score >= SIMILARITY_THRESHOLD:
        keyword, main_cat, sub_cat = occupation_meta[top_idx]
        return {
            "matched_keyword": keyword,
            "main_category": main_cat,
            "sub_category": sub_cat,
            "similarity": round(top_score, 4)
        }
    else:
        return None

# --- Streamlit UI ---
st.title("Twitter Bio Analyzer")
st.write("""
Paste a Twitter bio below to extract trading/crypto entities and get the closest-matching occupation.
""")

# --- Single Bio Input ---
st.subheader("Analyze a Single Twitter Bio")

bio_input = st.text_area("Paste a Twitter bio here", height=100)
analyze_button = st.button("Analyze Bio")

if analyze_button and bio_input.strip():
    with st.spinner("Processing..."):
        entities = gliner_model.predict_entities(bio_input, ["brand", "stock", "cryptocurrency"])
        occupation_match = match_occupation_semantic(bio_input)

        st.markdown("### 🧾 Extracted Entities")
        if entities:
            for ent in entities:
                st.write(f"- **{ent['text']}** → `{ent['label']}`")
        else:
            st.info("No entities found.")

        st.markdown("### Matched Occupation")
        if occupation_match:
            st.write(f"**Keyword:** {occupation_match['matched_keyword']}")
            st.write(f"**Main Category:** {occupation_match['main_category']}")
            st.write(f"**Similarity Score:** {occupation_match['similarity']}")
        else:
            st.warning("No occupation match found.")


import pandas as pd

st.subheader("Analyze Multiple Bios from File")

uploaded_file = st.file_uploader("Upload a .txt or .csv file", type=["txt", "csv"])

if uploaded_file is not None:
    with st.spinner("Processing file..."):
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                if "bio" not in df.columns:
                    st.error("CSV file must have a column named 'bio'")
                else:
                    bios = df["bio"].dropna().tolist()
            else:
                bios = uploaded_file.read().decode("utf-8").splitlines()
                bios = [line.strip() for line in bios if line.strip()]

            results = []

            for bio in bios:
                entities = gliner_model.predict_entities(bio, ["brand", "stock", "cryptocurrency"])
                occupation_match = match_occupation_semantic(bio)

                results.append({
                    "Bio": bio,
                    "Entities": ", ".join(f"{e['text']} → {e['label']}" for e in entities) if entities else "None",
                    "Occupation": occupation_match["matched_keyword"] if occupation_match else "None",
                    "Category": occupation_match["main_category"] if occupation_match else "None",
                    "Score": occupation_match["similarity"] if occupation_match else None
                })

            st.success(f"Processed {len(results)} bios.")
            st.dataframe(pd.DataFrame(results))

        except Exception as e:
            st.error(f"Error processing file: {e}")
