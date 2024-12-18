from sklearn.preprocessing import OneHotEncoder
import pandas as pd


def data_normalizer(df: pd.DataFrame, column_name_to_normalize: list, scaler):
    scaler = scaler.fit(df[column_name_to_normalize])
    nrm_arr = scaler.transform(df[column_name_to_normalize])
    df_nrm_selected_columns = pd.DataFrame(
        nrm_arr, columns=df[column_name_to_normalize].columns, index=df.index
    )
    remaining_df = df.drop(columns=column_name_to_normalize)
    df_nrm = pd.concat([remaining_df, df_nrm_selected_columns], axis=1)

    return df_nrm, scaler


def categorical_encoder(df: pd.DataFrame, column_name_to_encode: list):
    """
    This function encodes desired data columns `column_name_to_encode`
    using one-hot-encoding method
    Returns: modified df -> note the function will REPLACE the
             original data (instead of creating new columns)

    """
    encoder = OneHotEncoder()
    encoded_arr = encoder.fit_transform(df[column_name_to_encode])
    encoded_df = pd.DataFrame(
        encoded_arr.toarray(),
        columns=encoder.get_feature_names_out(column_name_to_encode),
        index=df.index,
    )
    remaining_df = df.drop(columns=column_name_to_encode)
    df = pd.concat(
        [remaining_df, encoded_df], axis=1
    )

    return df
