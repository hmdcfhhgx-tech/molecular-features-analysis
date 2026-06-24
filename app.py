import streamlit as st
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.preprocessing import StandardScaler
import joblib

st.set_page_config(page_title="متنبئ جهد الأكسدة والاختزال", layout="centered")

@st.cache_resource
def load_model():
    model = joblib.load('redox_model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

model, scaler = load_model()

def compute_molecular_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024)
    fp_array = np.array(fingerprint)
    
    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumHeteroatoms(mol),
        Descriptors.RingCount(mol),
        rdMolDescriptors.CalcNumSaturatedRings(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
        Descriptors.MaxPartialCharge(mol),
        Descriptors.MinPartialCharge(mol),
        Descriptors.MolLogP(mol),
        Descriptors.HeavyAtomCount(mol)
    ]
    
    return np.concatenate([fp_array, descriptors]), mol

st.title("متنبئ جهد الأكسدة والاختزال")
st.markdown("أدخل صيغة SMILES لجزيء عضوي للحصول على الجهد الكهروكيميائي المتوقع")

examples = {
    "اختر مثالاً...": "",
    "بنزوكينون": "O=C1C=CC(=O)C=C1",
    "نفثوكينون": "O=C1C=CC(=O)C2=C1C=CC=C2",
    "أنثراكينون": "O=C1C=CC(=O)C3=C1C=CC2=C3C=CC=C2",
    "رباعي فلورو-بنزوكينون": "O=C1C(F)=C(F)C(=O)C(F)=C1F",
}

selected = st.selectbox("اختر مثالاً جاهزاً:", list(examples.keys()))
smiles_input = st.text_input(
    "أو أدخل صيغة SMILES مباشرة:",
    value=examples[selected] if selected != "اختر مثالاً..." else ""
)

if st.button("توقع الجهد", type="primary"):
    if smiles_input:
        result = compute_molecular_features(smiles_input)
        if result is not None:
            features, mol = result
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)[0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("الجهد المتوقع", f"{prediction:.3f} V")
            with col2:
                st.metric("نوع الجزيء", 
                         "موجب (أنود)" if prediction > -0.5 else "سالب (كاثود)")
            
            try:
                from rdkit.Chem import Draw
                img = Draw.MolToImage(mol, size=(300, 200))
                st.image(img, caption="تركيب الجزيء")
            except Exception:
                st.info("صيغة SMILES صالحة")
        else:
            st.error("صيغة SMILES غير صالحة")
    else:
        st.warning("الرجاء إدخال صيغة SMILES")
