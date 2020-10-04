import click
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


@click.command()
@click.option('-i', prompt='Input CSV file path', help='Input CSV file path.')
@click.option('-t', prompt='Topics Column name', help='The topics column name.')
@click.option('-o', prompt='Output CSV file path', help='Output CSV file path.')
def convert(i, t, o):
    """Convert a csv file with comma separated topics to multi-hot encoded"""

    df = pd.read_csv(i)

    topics_col = t
    df[topics_col] = df[topics_col].astype(str)

    df[topics_col] = df[topics_col].apply(lambda x: x.split(','))
    mlb = MultiLabelBinarizer()
    one_hot_encoded_train = mlb.fit_transform(df[topics_col])
    df = df.join(pd.DataFrame(one_hot_encoded_train, columns=mlb.classes_, index=df.index))
    print("Classes: ", mlb.classes_)

    df = df.rename(columns={'label': 'label_tag'})
    df = df.rename(columns={'text': 'text_tag'})

    df['labels'] = pd.Series(list(one_hot_encoded_train), index=df.index)

    df.to_csv(o, index=False)


if __name__ == '__main__':
    convert()
