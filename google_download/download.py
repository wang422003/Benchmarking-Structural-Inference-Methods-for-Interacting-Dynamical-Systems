import sys
import requests
import re
import os
import tqdm
import zipfile

# import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()

download_dict = {
    "BN_springs.zip": "https://drive.google.com/file/d/1XEBnQpkizJ4snErhFXgYPbn09seyYngB/view?usp=sharing",
    "CRNA_springs.zip": "https://drive.google.com/file/d/1LqpCTYF3GUmQ4E-p4eENJOyIbGbXpHj9/view?usp=sharing",
    "FW_springs.zip": "https://drive.google.com/file/d/1WNbY4IasNDmUQVZdc8kVaeV9G6iwKAy6/view?usp=sharing",
    "GCN_springs.zip": "https://drive.google.com/file/d/1Xa5wfKexz2nsPeIzUUyDT-T5ioAXfA8w/view?usp=sharing",
    "GRN_springs.zip": "https://drive.google.com/file/d/1c-89YgmoBpFUfuMC-7RXp3Z3LwjTUVpV/view?usp=sharing",
    "IN_springs.zip": "https://drive.google.com/file/d/1Vr0Tzg-b73qVB8xnokCFkl16rXq0_8Wt/view?usp=sharing",
    "LN_springs.zip": "https://drive.google.com/file/d/1aND1X_mycAAv835DMf88tYwiMVwuY7Mp/view?usp=sharing",
    "MMO_springs.zip": "https://drive.google.com/file/d/1bNvDlXNFleC51R4HC-V37l24llkrZwe1/view?usp=sharing",
    "RNLO_springs.zip": "https://drive.google.com/file/d/1DkokkVn_WyarJQJ22lngZQLXA3TzlQ6Q/view?usp=sharing",
    "SN_springs.zip": "https://drive.google.com/file/d/1VAQc_QwOPIkDT0hYS1FlCGQIf9SFLd58/view?usp=sharing",
    "VN_springs.zip": "https://drive.google.com/file/d/1czzmCG5dHhp-rJ6lVxwRSYxlSutbPkyy/view?usp=sharing",

    "BN_netsims.zip": "https://drive.google.com/file/d/1A9F8HSjTR3yjlfUXzlyY_rWPNNejlKb7/view?usp=sharing",
    "CRNA_netsims.zip": "https://drive.google.com/file/d/1KGO7jTIujAf6vEfPqQqvjoHPZ4Uq8VmZ/view?usp=sharing",
    "FW_netsims.zip": "https://drive.google.com/file/d/1phPHRsObX1nxgLQQVp57Wu3GlKF2ebo4/view?usp=sharing",
    "GCN_netsims.zip": "https://drive.google.com/file/d/1jW9JkNYkKYf-kk44ZwFF12CFPib_FfDc/view?usp=sharing",
    "GRN_netsims.zip": "https://drive.google.com/file/d/1aoh69rWsC1zZmBpv5NTQaIAStGb3WBO8/view?usp=sharing",
    "IN_netsims.zip": "https://drive.google.com/file/d/1vSbKHIUSrdFJ9Af34O1OkglF1kcYajuE/view?usp=sharing",
    "LN_netsims.zip": "https://drive.google.com/file/d/1Tbxseb1O30a6fo7pIAD7F-aJ2mV5xb_m/view?usp=sharing",
    "MMO_netsims.zip": "https://drive.google.com/file/d/10jL2rKFyH5Gh2yQV5jKvParahWRFe2Eb/view?usp=sharing",
    "RNLO_netsims.zip": "https://drive.google.com/file/d/1etznIHvYsfWRWGjhbsQ6F7jDtksdY722/view?usp=sharing",
    "SN_netsims.zip": "https://drive.google.com/file/d/1FWvfoRk9D4yZklCSEPAGJ_RToFClAgMc/view?usp=sharing",
    "VN_netsims.zip": "https://drive.google.com/file/d/1z9pM4R-ZvFcmLwnMqlfgOruEBL3Tur3n/view?usp=sharing",

    "BN_netsims_N1.zip": "https://drive.google.com/file/d/1Hc8Dxg6Ns2iRBqB_jZI9X7G4w6b2EYkO/view?usp=sharing",
    "BN_netsims_N2.zip": "https://drive.google.com/file/d/1fZgepnFgQfXViVRa8eq9hlX_Yavycj5Z/view?usp=sharing",
    "BN_netsims_N3.zip": "https://drive.google.com/file/d/1rdvfK5HsZU6kXvX0Qc92sEkZRpLmM1im/view?usp=sharing",
    "BN_netsims_N4.zip": "https://drive.google.com/file/d/1ME_twen06maYNSfa40XmvRbSJ_f2H7SG/view?usp=sharing",
    "BN_netsims_N5.zip": "https://drive.google.com/file/d/1aJ0HvWp7s_IKF6NFAngEVOymwS6DmTA5/view?usp=sharing",
}

unzip_dict = {
    "BN_springs.zip": "src/simulations/brain_networks/directed/springs",
    "CRNA_springs.zip": "src/simulations/chemical_reaction_networks_in_atmosphere/directed/springs",
    "FW_springs.zip": "src/simulations/food_webs/directed/springs",
    "GCN_springs.zip": "src/simulations/gene_coexpression_networks/undirected/springs",
    "GRN_springs.zip": "src/simulations/gene_regulatory_networks/directed/springs",
    "IN_springs.zip": "src/simulations/intercellular_networks/directed/springs",
    "LN_springs.zip": "src/simulations/landscape_networks/undirected/springs",
    "MMO_springs.zip": "src/simulations/man-made_organic_reaction_networks/directed/springs",
    "RNLO_springs.zip": "src/simulations/reaction_networks_inside_living_organism/directed/springs",
    "SN_springs.zip": "src/simulations/social_networks/directed/springs",
    "VN_springs.zip": "src/simulations/vascular_networks/directed/springs",

    "BN_netsims.zip": "src/simulations/brain_networks/directed/netsims",
    "CRNA_netsims.zip": "src/simulations/chemical_reaction_networks_in_atmosphere/directed/netsims",
    "FW_netsims.zip": "src/simulations/food_webs/directed/netsims",
    "GCN_netsims.zip": "src/simulations/gene_coexpression_networks/undirected/netsims",
    "GRN_netsims.zip": "src/simulations/gene_regulatory_networks/directed/netsims",
    "IN_netsims.zip": "src/simulations/intercellular_networks/directed/netsims",
    "LN_netsims.zip": "src/simulations/landscape_networks/undirected/netsims",
    "MMO_netsims.zip": "src/simulations/man-made_organic_reaction_networks/directed/netsims",
    "RNLO_netsims.zip": "src/simulations/reaction_networks_inside_living_organism/directed/netsims",
    "SN_netsims.zip": "src/simulations/social_networks/directed/netsims",
    "VN_netsims.zip": "src/simulations/vascular_networks/directed/netsims",

    "BN_netsims_N1.zip": "src/simulations/brain_networks/directed/netsims",
    "BN_netsims_N2.zip": "src/simulations/brain_networks/directed/netsims",
    "BN_netsims_N3.zip": "src/simulations/brain_networks/directed/netsims",
    "BN_netsims_N4.zip": "src/simulations/brain_networks/directed/netsims",
    "BN_netsims_N5.zip": "src/simulations/brain_networks/directed/netsims",
}

zip_file_error = []

def download_file_from_google_drive(file_id, destination):
    session = requests.Session()

    response = session.get("https://docs.google.com/uc?export=download&confirm=t", params={"id": file_id}, stream=True)
    # token = get_confirm_token(response)
    cookie = re.findall(r'type="hidden"\W+name="(.*?)\W+value="(.*?)">', response.text)
    cookie = {i[0]: i[1] for i in cookie}

    # if token:
    # if True:
    params = {"id": file_id, "acknowledgeAbuse": True}
    params.update(cookie)
    response = session.get("https://drive.usercontent.google.com/download", params=params, stream=True)
    size=int(response.headers["Content-Length"])

    save_response_content(response, destination, size)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination, size=0):
    CHUNK_SIZE = 32768
    bar = tqdm.tqdm(total=size, unit='B', unit_scale=True, unit_divisor=1024)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                bar.update(len(chunk))


def unzip(from_, to):
    try:
        with zipfile.ZipFile(from_, 'r') as zip_ref:
            print(f"extracting {from_} to {to}")
            zip_ref.extractall(to)
    except zipfile.BadZipFile:
        print(f"Error in {from_}.")
        zip_file_error.append(from_)


def main():
    for destination_short, file_id in download_dict.items():
        file_id = re.findall(r"/d/(.*?)/view", file_id)[0]
        destination = destination_short
        unzip_destination = os.path.join(os.path.dirname(sys.argv[0]), "..", *unzip_dict[destination_short].split("/"))
        unzip_destination = os.path.normpath(unzip_destination)
        destination = os.path.join(os.path.dirname(sys.argv[0]), destination)

        if os.path.exists(destination) and len(os.listdir(unzip_destination)) > 10:
            print(f"{destination_short}\t already exists and is unzipped. Skipping download.")
        elif os.path.exists(destination) and len(os.listdir(unzip_destination)) <= 10:
            print(f"{destination_short}\t already exists but is not unzipped. Unzipping.")
            unzip(destination, unzip_dict[destination_short])
        elif not os.path.exists(destination) and len(os.listdir(unzip_destination)) > 10:
            print(f"{destination_short}\t already extracted properly. Skipping download.")
            continue
        else:
            print(f"{destination_short}\t does not exist. Downloading {file_id} to {destination_short}.")
            download_file_from_google_drive(file_id, destination)
            print("Unziping.")
            unzip(destination, unzip_dict[destination_short])
        
        print(f"Removing {destination}.")
        os.remove(destination)
    
    if len(zip_file_error) > 0:
        print("The following files could not be unzipped:")
        for file in zip_file_error:
            print(file)

if __name__ == "__main__":
    main()