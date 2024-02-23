
# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, avg, stddev, sum, count

# Create a Spark session
my_spark = SparkSession.builder.appName("CourseProcessing").getOrCreate()
print (my_spark)

"""# Data loading and exploration

### Load Coursera Course Dataset
"""

dataset_path = "/content/coursera_course_dataset.csv"

# Read dataset into DataFrame
df = my_spark.read.csv(dataset_path, header=True, inferSchema=True)

# Explore DataFrame structure and initial rows
df.printSchema()
df.show(5, truncate=False)

"""#	Data cleaning and preparation

### Extract and Cast Review Count
"""

# Extract the numeric review count from the "Review counts" column and cast it to an integer
df = df.withColumn(
    "ReviewCount",
    regexp_extract(col("Review counts"), r"\((\d+)K reviews\)", 1))

#cast it to an integer
df=df.withColumn("ReviewCount",df.ReviewCount.cast("integer"))

# Display the extracted review counts
df.select("ReviewCount").show()

df.printSchema()

df.select("Metadata").show()

"""### Expand Metadata Column"""

# Split Metadata column into Level, Certificate_Type, and Duration
df = df.withColumn('Level', split(df['Metadata'], ' · ')[0]) \
       .withColumn('Certificate_Type', split(df['Metadata'], ' · ')[1]) \
       .withColumn('Duration', split(df['Metadata'], ' · ')[2])

# Display expanded columns
df.select("Level", "Certificate_Type", "Duration").show()

# Print Dataframe Summary
df.show(5, truncate=False)
df.printSchema()

"""### Drop Unnecessary Columns and Review New DataFrame"""

new_df = df.drop(*["Metadata_Split", "Review counts", "Metadata"])

# Display and examine the updated DataFrame
new_df.show(5, truncate=False)
new_df.printSchema()

"""#	Descriptive and exploratory analysis

### Analyze Course Titles and Skills
"""

# Filter courses with high ratings
highly_rated_courses = new_df.filter(col('Ratings') >= 4.5)

print("Courses with Ratings Above 4.5:")
highly_rated_courses.show(10, truncate=False)

# Select relevant columns for analysis
title_skills_df = new_df.select(col("Title"), col("Skills"))

# Analyze word frequencies:
#   - Group by Title words and count occurrences
#   - Group by Skill words and count occurrences
#   - Display top 10 frequent words for each

title_word_counts = new_df.groupBy("Title").count().orderBy("count", ascending=False)
print("Top 10 frequent words in course titles:")
title_word_counts.show(10)


skill_word_counts = new_df.groupBy("Skills").count().orderBy("count", ascending=False)
print("Top 10 frequent words in course skills:")
skill_word_counts.show(10)

"""### Analyze Course Providers by Average Ratings"""

# Select relevant columns
organization_stats_df = new_df.select(col("Organization"), col("Ratings"), col("ReviewCount"), col("Certificate_Type"))

# Group by organization and calculate aggregate statistics
organization_comparison = organization_stats_df.groupBy("Organization").agg(
    avg("Ratings").alias("Average_Ratings"),
    sum("ReviewCount").alias("Total_ReviewCount"),
    count("Certificate_Type").alias("Certificate_Count")
)

# Order the results by average ratings
organization_comparison = organization_comparison.orderBy("Average_Ratings", ascending=False)

# Display the comparison results
organization_comparison.show(truncate=False)

"""### Analyze Correlations between Ratings and ReviewCount"""

# Select relevant columns
selected_df = new_df.select(col("Ratings"), col("ReviewCount"))

# Handle missing values
selected_df = selected_df.dropna()

# Calculate and display correlations
correlation_matrix = selected_df.select(corr("Ratings", "ReviewCount"))

# Display the correlation matrix
print("Correlation matrix:")
correlation_matrix.show()

"""### Analyze Course Duration by Organization and Level"""

# Select relevant columns (duration, organization, level) and clean duration data
selected_DOL_df = new_df.select(col("Duration"), col("Organization"), col("Level"))
selected_DOL_df = selected_DOL_df.withColumn("Duration", expr("regexp_replace(Duration, '[^0-9]+', '')").cast("integer"))

# Calculate and display duration statistics
duration_analysis = selected_DOL_df.groupBy("Organization", "Level").agg(
    avg("Duration").alias("Average_Duration")
)

# Show the duration analysis
duration_analysis.show(truncate=False)

"""### Analyze Course Difficulty Level Distribution"""

# Select and group by difficulty level
level_df = new_df.select(col("Level"))

level_distribution = level_df.groupBy("Level").agg(count("*").alias("Course_Count"))

# Display the distribution
level_distribution.show()

"""### Analyze Course Difficulty by Average Ratings and Review Counts"""

# Select relevant columns (difficulty level, ratings, review counts)
level_rating_review_df = new_df.select(col("Level"), col("Ratings"), col("ReviewCount"))

# Calculate and display average ratings and total review counts per difficulty
level_analysis = level_rating_review_df.groupBy("Level").agg(
    avg("Ratings").alias("Average_Ratings"),
    sum("ReviewCount").alias("Total_ReviewCount")
)

# Show the analysis results
level_analysis.show()

"""### Analyze Certificate Type Distribution by Organization"""

# Select relevant columns
certificate_org_df = new_df.select(col("Certificate_Type"), col("Organization"))

# Group by certificate type and organization, count occurrences
certificate_analysis = certificate_org_df.groupBy("Certificate_Type", "Organization").agg(
    count("*").alias("Certificate_Count")
)
# Order by certificate count in descending order
certificate_analysis = certificate_analysis.orderBy("Certificate_Count", ascending=False)

# Show the certificate type analysis
certificate_analysis.show(truncate=False)

"""### Analyze Total Certificates Issued by Each Organization"""

# Calculate and display total certificates per organization (Descending Order)
total_certificates_by_org = certificate_org_df.groupBy("Organization").agg(
    count("*").alias("Total_Certificates_Issued")
)

total_certificates_by_org = total_certificates_by_org.orderBy("Total_Certificates_Issued", ascending=False)

total_certificates_by_org.show(truncate=False)

