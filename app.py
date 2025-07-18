def get_defendant_charges_table(df):
    """
    Create a table showing each defendant with their charges
    Focus on defendants with 790.07 charges AND marijuana-related charges
    Exclude defendants with serious violent charges
    """
    # Define serious violent charges to exclude
    serious_charges = [
        'SHOOTING', 'MURDER', 'HOMICIDE', 'ASSAULT', 'BATTERY', 
        'ROBBERY', 'RAPE', 'KIDNAPPING', 'CARJACKING'
    ]
    
    # Define marijuana-related keywords
    marijuana_keywords = ['CANNABIS', 'MARIJUANA', 'THC', '893.13']
    
    # Get defendants with any 790.07 charge
    defendants_with_790_07 = df[df['Is_790_07_Any']]['Defendant_ID'].unique()
    
    # Get all charges for these defendants
    defendant_charges = df[df['Defendant_ID'].isin(defendants_with_790_07)]
    
    # Filter out defendants with serious charges
    defendants_to_exclude = set()
    for defendant_id in defendants_with_790_07:
        charges = defendant_charges[defendant_charges['Defendant_ID'] == defendant_id]
        
        # Check if any charge contains serious keywords
        for _, charge in charges.iterrows():
            desc = str(charge['ChargeOffenseDescription']).upper() if pd.notna(charge['ChargeOffenseDescription']) else ''
            statute_desc = str(charge['Statute_Description']).upper() if pd.notna(charge['Statute_Description']) else ''
            
            if any(keyword in desc for keyword in serious_charges) or \
               any(keyword in statute_desc for keyword in serious_charges):
                defendants_to_exclude.add(defendant_id)
                break
    
    # Filter to only defendants with marijuana-related charges
    defendants_with_marijuana = set()
    for defendant_id in defendants_with_790_07:
        if defendant_id in defendants_to_exclude:
            continue
            
        charges = defendant_charges[defendant_charges['Defendant_ID'] == defendant_id]
        
        # Check if any charge contains marijuana keywords
        for _, charge in charges.iterrows():
            desc = str(charge['ChargeOffenseDescription']).upper() if pd.notna(charge['ChargeOffenseDescription']) else ''
            statute_desc = str(charge['Statute_Description']).upper() if pd.notna(charge['Statute_Description']) else ''
            statute = str(charge['Statute']).upper() if pd.notna(charge['Statute']) else ''
            
            if any(keyword in desc for keyword in marijuana_keywords) or \
               any(keyword in statute_desc for keyword in marijuana_keywords) or \
               any(keyword in statute for keyword in marijuana_keywords):
                defendants_with_marijuana.add(defendant_id)
                break
    
    # Create a pivot-style view
    defendant_summary = []
    
    for defendant_id in list(defendants_with_marijuana)[:50]:  # Limit to first 50 for performance
        charges = defendant_charges[defendant_charges['Defendant_ID'] == defendant_id].sort_values('OffenseDate')
        
        # Get basic info
        officer = charges['Lead_Officer'].mode()[0] if len(charges) > 0 else 'Unknown'
        
        # Get charge list
        charge_list = charges['Statute'].tolist()
        
        # Create row
        row = {
            'Defendant': defendant_id,
            'Officer': officer,
            'Total_Charges': len(charges),
            'Charges': ', '.join(charge_list[:5])  # Show first 5 charges
        }
        
        # Add specific charge columns with descriptions
        for i, (_, charge) in enumerate(charges.iterrows()):
            if i < 5:  # Limit to 5 charges for display
                # Get the description, truncate if too long
                desc = charge['ChargeOffenseDescription'] if pd.notna(charge['ChargeOffenseDescription']) else charge['Statute_Description']
                if len(str(desc)) > 50:
                    desc = str(desc)[:50] + '...'
                
                date_str = charge['OffenseDate'].strftime('%m/%d/%Y') if pd.notna(charge['OffenseDate']) else 'N/A'
                row[f'Charge_{i+1}'] = f"{desc} ({date_str})"
        
        defendant_summary.append(row)
    
    return pd.DataFrame(defendant_summary)
