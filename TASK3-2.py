{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Import pandas and file( here a cvs file)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "file = pandas.read_csv(file path\") "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### view file add a number for nbr of lines"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "file.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### choose wich columns you want and filter"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = file[['column1', 'column2', 'Column3']]\n",
    "\n",
    "def text_category(p):\n",
    "    if conition:\n",
    "        return -----\n",
    "    elif p==0:\n",
    "        return ------l\n",
    "    else:\n",
    "        return ------\n",
    "\n",
    "df = df.drop(df[df[condition].index)\n",
    "\n",
    "# convert a column to numeric for example                \n",
    "df[\"column\"]= pd.to_numeric(df['df'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Visualise datas ( barplot, piechart, wordcloud)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "#barplot\n",
    "plt.figure()\n",
    "plt.bar(df[\"columns\"].value_counts().index, height= df[\"columns2\"].value_counts().values);\n",
    "#piechart\n",
    "plt.pie(df[\"columns\"].value_counts());\n",
    "\n",
    "#wordcloud\n",
    "text = list(df[\"columns\"])\n",
    "text2=str(text)\n",
    "\n",
    "# Create and generate a word cloud image:\n",
    "drop_words = ['gt', 'gt', 'ki'] \n",
    "wordcloud = WordCloud(background_color=\"black\", stopwords= drop_words ).generate(text2)\n",
    "\n",
    "# Display the generated image:\n",
    "plt.imshow(wordcloud, interpolation='bilinear')\n",
    "plt.axis(\"off\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Mapping and defining variables"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"column\"]=df[\"column\"].map({\"positive\":1, \"negative\":0})\n",
    "\n",
    "x=df[\"column\"]\n",
    "y=df[\"column1\"]\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Split datas"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(x, y,\n",
    "                                                   test_size=0.40,\n",
    "                                                   random_state=2021)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Vectorise"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "vectorizer = CountVectorizer()\n",
    "X_train_vec = vectorizer.fit_transform(X_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Clustering datas"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.cluster import KMeans\n",
    "import numpy as np\n",
    "kmeans = KMeans(n_clusters=10, random_state=0).fit(X_train_vec)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Making  an SGDClassifier model "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from sklearn.linear_model import SGDClassifier\n",
    "\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "model= SGDClassifier()\n",
    "model.fit(X_train_vec, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.intercept_\n",
    "\n",
    "from sklearn.metrics import confusion_matrix\n",
    "\n",
    "Accuracy"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
