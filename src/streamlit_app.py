import streamlit as st

import streamlit as st


def main():
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 90% !important;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.title("🏠 House Prices Prediction App")
    st.subheader("Property Data Input Form")

    # Use columns to display fields side by side
    col1, col2 = st.columns(2)

    with col1:
        # MSSubClass
        st.subheader("MSSubClass")
        ms_subclass = st.selectbox(
            "Type of dwelling involved in the sale:",
            ["20 - 1-STORY 1946 & NEWER ALL STYLES", "30 - 1-STORY 1945 & OLDER",
             "40 - 1-STORY W/FINISHED ATTIC ALL AGES", "45 - 1-1/2 STORY - UNFINISHED ALL AGES",
             "50 - 1-1/2 STORY FINISHED ALL AGES", "60 - 2-STORY 1946 & NEWER",
             "70 - 2-STORY 1945 & OLDER", "75 - 2-1/2 STORY ALL AGES",
             "80 - SPLIT OR MULTI-LEVEL", "85 - SPLIT FOYER", "90 - DUPLEX - ALL STYLES AND AGES",
             "120 - 1-STORY PUD - 1946 & NEWER", "150 - 1-1/2 STORY PUD - ALL AGES",
             "160 - 2-STORY PUD - 1946 & NEWER", "180 - PUD - MULTILEVEL",
             "190 - 2 FAMILY CONVERSION"]
        )

        # LotFrontage
        st.subheader("LotFrontage")
        lot_frontage = st.number_input("Linear feet of street connected to property:", min_value=0, step=1)

        # Street
        st.subheader("Street")
        street = st.selectbox("Type of road access to property:", ["Grvl - Gravel", "Pave - Paved"])

        # LotShape
        st.subheader("LotShape")
        lot_shape = st.selectbox("General shape of property:", ["Reg - Regular", "IR1 - Slightly irregular",
                                                                "IR2 - Moderately Irregular", "IR3 - Irregular"])

        # Utilities
        st.subheader("Utilities")
        utilities = st.selectbox("Type of utilities available:",
                                 ["AllPub - All public Utilities", "NoSewr - Electricity, Gas, Water (Septic Tank)",
                                  "NoSeWa - Electricity and Gas Only", "ELO - Electricity only"])

        # LandSlope
        st.subheader("LandSlope")
        land_slope = st.selectbox("Slope of property:",
                                  ["Gtl - Gentle slope", "Mod - Moderate Slope", "Sev - Severe Slope"])

    with col2:
        # MSZoning
        st.subheader("MSZoning")
        ms_zoning = st.selectbox("General zoning classification of the sale:",
                                 ["A - Agriculture", "C - Commercial", "FV - Floating Village Residential",
                                  "I - Industrial", "RH - Residential High Density", "RL - Residential Low Density",
                                  "RP - Residential Low Density Park", "RM - Residential Medium Density"]
                                 )

        # LotArea
        st.subheader("LotArea")
        lot_area = st.number_input("Lot size in square feet:", min_value=0, step=1)

        # Alley
        st.subheader("Alley")
        alley = st.selectbox("Type of alley access to property:",
                             ["NA - No alley access", "Grvl - Gravel", "Pave - Paved"])

        # LandContour
        st.subheader("LandContour")
        land_contour = st.selectbox("Flatness of the property:",
                                    ["Lvl - Near Flat/Level", "Bnk - Banked", "HLS - Hillside", "Low - Depression"])

        # LotConfig
        st.subheader("LotConfig")
        lot_config = st.selectbox("Lot configuration:",
                                  ["Inside - Inside lot", "Corner - Corner lot", "CulDSac - Cul-de-sac",
                                   "FR2 - Frontage on 2 sides", "FR3 - Frontage on 3 sides"])

        # Neighborhood
        st.subheader("Neighborhood")
        neighborhood = st.selectbox("Physical locations within Ames city limits:",
                                    ["Blmngtn - Bloomington Heights", "Blueste - Bluestem", "BrDale - Briardale",
                                     "BrkSide - Brookside", "ClearCr - Clear Creek", "CollgCr - College Creek",
                                     "Crawfor - Crawford", "Edwards - Edwards", "Gilbert - Gilbert",
                                     "IDOTRR - Iowa DOT and Rail Road", "MeadowV - Meadow Village",
                                     "Mitchel - Mitchell", "Names - North Ames", "NoRidge - Northridge",
                                     "NPkVill - Northpark Villa", "NridgHt - Northridge Heights",
                                     "NWAmes - Northwest Ames", "OldTown - Old Town", "SWISU - South & West of ISU",
                                     "Sawyer - Sawyer", "SawyerW - Sawyer West", "Somerst - Somerset",
                                     "StoneBr - Stone Brook", "Timber - Timberland", "Veenker - Veenker"])

    # YearBuilt and YearRemodAdd
    st.subheader("YearBuilt and YearRemodAdd")
    year_built = st.number_input("Original construction date:", min_value=1800, max_value=2024, step=1)
    year_remod = st.number_input("Remodel date (same as construction date if no remodeling):",
                                 min_value=1800, max_value=2024, step=1)

    # Submit button
    if st.button("Submit"):
        st.success("Data Submitted Successfully!")
        st.write("Here is the data you submitted:")
        st.write({
            "MSSubClass": ms_subclass,
            "MSZoning": ms_zoning,
            "LotFrontage": lot_frontage,
            "LotArea": lot_area,
            "Street": street,
            "Alley": alley,
            "LotShape": lot_shape,
            "LandContour": land_contour,
            "Utilities": utilities,
            "LotConfig": lot_config,
            "LandSlope": land_slope,
            "Neighborhood": neighborhood,
            "YearBuilt": year_built,
            "YearRemodAdd": year_remod
        })


if __name__ == "__main__":
    main()
