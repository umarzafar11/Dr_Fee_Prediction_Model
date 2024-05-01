from flask import Flask , render_template,request
import joblib
import numpy as np
import lightgbm
from xgboost import XGBRegressor

#Getting saved Model
model = joblib.load('dr_pythmodel_xgb.joblib', 'rb')

app = Flask(__name__)
@app.route('/')

def index():
    #return 'Hello World'

    return render_template('index.html')

@app.route("/predict", methods = ["GET",'POST'])
def predict_fee():


    exp = int(request.form.get('exp'))
    #specs = int(request.form.get('specs'))
    reviews = int(request.form.get('reviews'))
    sat_rate = int(request.form.get('sat_rate'))
    #province = int(request.form.get('province'))
    #city = int(request.form.get('city'))
    avg_time = int(request.form.get('avg_time'))
    wait_time = int(request.form.get('wait_time'))
    #d_specs = request.form.get('specsDropdown')
    #d_city = request.form['cityDropdown']
    #d_province = request.form['provinceDropdown']

    #print(d_city)
    #print(d_province)

    my_specs = {'dermatologist': 2, 'neurologist': 19, 'gynecologist': 11, 'urologist': 31, 'gastroenterologist': 9,
             'pulmonologist': 26, 'orthopedic-surgeon': 20, 'pediatrician': 24, 'general-physician': 10,
             'nephrologist': 15, 'neuro-psychiatrist': 17, 'neuro-physician': 16, 'endocrinologist': 3,
             'sexologist': 29, 'ent-specialist': 5, 'eye-surgeon': 7, 'neuro-surgeon': 18, 'andrologist': 0,
             'neonatologist': 14, 'spinal-surgeon': 30, 'pediatric-gastroenterologist': 21, 'internal-medicine': 12,
             'family-medicine': 8, 'ent-surgeon': 6, 'psychiatrist': 25, 'endourologist': 4,
             'pediatric-neuro-physician': 23, 'pediatric-nephrologist': 22, 'medical-specialist': 13,
             'regenerative-medicine-specialist': 27, 'rheumatologist': 28, 'cardiologist': 1}

    my_province = {'Punjab': 3, 'KPK': 2, 'Sindh': 4, 'Capital Territory': 1, 'Balochistan': 0, 'Unknown': 5}

    my_city = {'QUETTA': 87, 'ABBOTTABAD': 0, 'ALIPUR': 1, 'ATTOCK': 2, 'BADEN': 3, 'BAHAWALNAGAR': 4, 'BAHAWALPUR': 5,
            'BAJAUR-AGENCY': 6, 'BANNU': 7, 'BHAKKAR': 8, 'BHALWAL': 9, 'BUNER': 10, 'BUREWALA': 11, 'CHAKWAL': 12,
            'CHAMAN': 13, 'CHARSADDA': 14, 'CHICHAWATNI': 15, 'CHINIOT': 16, 'CHISHTIAN': 17, 'DARGAI': 18, 'DASKA': 19,
            'DERA-GHAZI-KHAN': 20, 'DERA-ISMAIL-KHAN': 21, 'DIJKOT': 22, 'DINGA': 23, 'DUNYAPUR': 24, 'GILGIT': 26,
            'GOJRA': 27, 'GUJAR-KHAN': 28, 'GUJRAT': 30, 'HAFIZABAD': 31, 'HANGU': 32, 'HARIPUR': 33, 'HYDERABAD': 34,
            'ISTANBUL': 36, 'IZMIR': 37, 'JACOBABAD': 38, 'JAMSHORO': 39, 'JARANWALA': 40, 'JAUHARABAD': 41,
            'JHANG': 42,
            'JHELUM': 43, 'KABIRWALA': 44, 'KAMOKE': 45, 'KANDIARO': 46, 'KASHMOR': 48, 'KASUR': 49, 'KHAIRPUR': 50,
            'KHANEWAL': 51, 'KHANPUR': 52, 'KHARIAN': 53, 'KHUSHAB': 54, 'KHUZDAR': 55, 'KOHAT': 56, 'KOT-ADDU': 57,
            'KOTLI': 58, 'LALAMUSA': 60, 'LARKANA': 61, 'LAYYAH': 62, 'LODHRAN': 63, 'LORALAI': 64, 'MALAKAND': 65,
            'MANDI-BAHAUDDIN': 66, 'MANSEHRA': 67, 'MARDAN': 68, 'MATIARI': 69, 'MIAN-CHANNU': 70, 'MIANWALI': 71,
            'MIRPUR': 72, 'MIRPUR-KHAS': 73, 'MITHI': 74, 'MURIDKE': 76, 'MUZAFFAR-GARH': 77, 'NANKANA-SAHIB': 78,
            'NAROWAL': 79, 'NAWABSHAH': 80, 'NOWSHERA': 81, 'OKARA': 82, 'PAKPATTAN': 83, 'PASRUR': 84, 'PATTOKI': 85,
            'RAHIM-YAR-KHAN': 88, 'RAJAN-PUR': 89, 'RAWALAKOT': 90, 'RENALA-KHURD': 91, 'RIYADH': 92, 'SADIQABAD': 93,
            'SAHIWAL': 94, 'SAMUNDRI': 95, 'SHAHDADPUR': 97, 'SHAHKOT': 98, 'SHEIKHUPURA': 99, 'SHORKOT': 100,
            'SKARDU': 102, 'SUKKUR': 103, 'SWABI': 104, 'SWAT': 105, 'TALAGANG': 106, 'TANDO-MUHAMMAD-KHAN': 107,
            'TAXILA': 108, 'THATTA': 109, 'TIMERGARA': 110, 'TOBA-TEK-SINGH': 111, 'TURBAT': 112, 'UMARKOT': 113,
            'VEHARI': 114, 'WAH-CANTT': 115, 'WAZIRABAD': 116, 'LAHORE': 59, 'MULTAN': 75, 'PESHAWAR': 86,
            'SIALKOT': 101,
            'GUJRANWALA': 29, 'SARGODHA': 96, 'FAISALABAD': 25, 'ISLAMABAD': 35, 'KARACHI': 47}


    #selected_specialization = None
    #selected_province = None
    #selected_city = None
    # Get selected values from dropdowns
    specs_dropdown = request.form.get('specsDropdown')
    province_dropdown = request.form.get('provinceDropdown')
    city_dropdown = request.form.get('cityDropdown')

    # Print selected values in console
    print("Selected Specialization:", specs_dropdown)
    print("Selected Province:", province_dropdown)
    print("Selected City:", city_dropdown)
    #print(type(specs_dropdown))
    #print(type(province_dropdown))
    #print(type(city_dropdown))
    specs = my_specs.get(specs_dropdown)
    province = my_province.get(province_dropdown)
    city = my_city.get(city_dropdown)

    #Getting Model Results
    result = model.predict(np.array([[exp,specs,reviews,sat_rate,province,city,avg_time,wait_time]]))
    print(result)


    print(province)


    predict_fee_dr = 'Predicted Fee of a '+specs_dropdown+' in '+city_dropdown+' having given criteria is '+str(result)
    predict_fee = model.predict()
    #return render_template('index.html',result = result)
    # Showing result in the same webpage
    return str()




