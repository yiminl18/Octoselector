"""Extract a fixed-order feature vector from an SQL query and its paired natural-language question.
Features capture query structure (tables, joins, predicates, grouping/ordering/limits, aggregates, nested queries),
and simple NL length/keyword signals, for downstream NL2SQL models (e.g., RF regressors, clustering)."""

import re

def extract_features(sql_query,NLquery):
    # Preprocess SQL (remove comments, extra whitespace, and convert to uppercase)
    sql_query_clean = preprocess_sql(sql_query)
    
    # Compute each feature as defined:
    num_tables = get_tables_count(sql_query_clean)
    num_predicates = get_predicates_count(sql_query_clean)
    num_nested = get_nested_queries_count(sql_query_clean)
    # bool_recursive = get_recursive_bool(sql_query_clean)
    num_sql_concepts = get_sql_concepts_count(sql_query_clean)
    num_groupby = get_group_by_num(sql_query_clean)
    bool_orderby = get_order_by_bool(sql_query_clean)
    bool_limit = get_limit_bool(sql_query_clean)
    bool_distinct = get_select_distinct_bool(sql_query_clean)
    num_agg = get_aggregation_count(sql_query_clean)
    num_projected_attributes = get_projected_attributes_num(sql_query_clean)
    num_joins = get_num_joins(sql_query_clean)
    NLquery_len_nomalized = get_NLquery_len_normalized(NLquery,normalized=True)
    num_logical_operators = get_num_logical_operators(sql_query_clean)
    num_having_conditions = get_num_having_conditions(sql_query_clean)
    bool_uses_like = get_bool_uses_like(sql_query_clean)
    num_involved_columns = get_num_involved_columns(sql_query_clean)
    # Feature vector order:
    # [#tables, #predicates, #nested queries, bool_recursive, #sql_concepts,
    #  #groupBy, bool_orderBy, bool_Limit, bool_Distinct, #aggregation operators, #projected attributes]
    return [num_tables, num_predicates, num_nested, num_sql_concepts,
            num_groupby, bool_orderby, bool_limit, bool_distinct, num_agg, num_projected_attributes,
            num_joins, NLquery_len_nomalized, num_logical_operators, num_having_conditions,
            bool_uses_like, num_involved_columns]

    ## considering add more features:
     ## [num_joins, NLquery_len_nomalized, num_logical_operators, num_having_conditions]
### Helper functions ###

def preprocess_sql(sql_query):
    # Remove single-line comments
    sql_query = re.sub(r'--.*?$', '', sql_query, flags=re.MULTILINE)
    # Remove multi-line comments
    sql_query = re.sub(r'/\*.*?\*/', '', sql_query, flags=re.DOTALL)
    # Normalize whitespace and convert to uppercase for case-insensitive matching
    sql_query = re.sub(r'\s+', ' ', sql_query).strip()
    return sql_query.upper()

def strip_strings(sql):
    # Remove string literals (single and double quotes) to avoid false positives
    sql = re.sub(r"'([^']*)'", '', sql)
    sql = re.sub(r'"([^"]*)"', '', sql)
    return sql

def split_sql_fields(field_str):
    """
    Splits a comma-separated SQL field string, taking care of parentheses.
    """
    fields = []
    bracket_level = 0
    current_field = ''
    for char in field_str:
        if char == '(':
            bracket_level += 1
            current_field += char
        elif char == ')':
            bracket_level -= 1
            current_field += char
        elif char == ',' and bracket_level == 0:
            fields.append(current_field.strip())
            current_field = ''
        else:
            current_field += char
    if current_field.strip():
        fields.append(current_field.strip())
    return fields

def remove_parentheses(s):
    # Remove content within parentheses for simpler splitting of predicates
    result = ''
    bracket_level = 0
    for char in s:
        if char == '(':
            bracket_level += 1
        elif char == ')':
            bracket_level -= 1
        elif bracket_level == 0:
            result += char
    return result

### Feature-specific functions ###

def get_tables_count(sql):
    """
    Count the number of tables referenced in the FROM clause.
    This includes tables mentioned in JOINs.
    """
    sql_no_strings = strip_strings(sql)
    # Look for FROM ... until one of these keywords appears
    match = re.search(r'\bFROM\b(.*?)(\bWHERE\b|\bGROUP BY\b|\bORDER BY\b|\bHAVING\b|\bLIMIT\b|$)', sql_no_strings, flags=re.DOTALL)
    tables = []
    if match:
        from_clause = match.group(1)
        # Split by commas and JOIN keywords.
        # First split by commas at top-level.
        candidates = split_sql_fields(from_clause)
        for cand in candidates:
            # Further split by JOIN (e.g., "TABLE1 INNER JOIN TABLE2 ON ..." -> get both TABLE1 and TABLE2)
            join_parts = re.split(r'\bJOIN\b', cand)
            for part in join_parts:
                # Take the first word which is likely the table name (ignoring aliases)
                tokens = part.strip().split()
                if tokens:
                    tables.append(tokens[0])
    # Remove duplicates and return count
    return len(set(tables))

def get_predicates_count(sql):
    """
    Count predicates from the WHERE clause and ON clauses in JOINs.
    """
    sql_no_strings = strip_strings(sql)
    predicate_count = 0

    # Count predicates in WHERE clause
    where_match = re.search(r'\bWHERE\b(.*?)(\bGROUP BY\b|\bORDER BY\b|\bHAVING\b|\bLIMIT\b|$)', sql_no_strings, flags=re.DOTALL)
    if where_match:
        where_clause = where_match.group(1)
        conditions = split_conditions(where_clause)
        predicate_count += len(conditions)
    
    # Also count predicates in JOIN ON clauses:
    on_matches = re.findall(r'\bON\b(.*?)(\bJOIN\b|\bWHERE\b|\bGROUP BY\b|\bORDER BY\b|\bHAVING\b|\bLIMIT\b|$)', sql_no_strings, flags=re.DOTALL)
    for match in on_matches:
        on_clause = match[0]
        conditions = split_conditions(on_clause)
        predicate_count += len(conditions)
    
    return predicate_count

def split_conditions(where_clause):
    """
    Splits a WHERE or ON clause into individual conditions based on AND/OR.
    """
    # Remove nested parentheses to avoid splitting inside them.
    simplified = remove_parentheses(where_clause)
    conditions = re.split(r'\bAND\b|\bOR\b', simplified)
    conditions = [cond.strip() for cond in conditions if cond.strip()]
    return conditions

def get_nested_queries_count(sql):
    """
    Count the number of nested queries (subqueries) inside the SQL.
    This searches for SELECT statements enclosed in parentheses.
    """
    sql_no_strings = strip_strings(sql)
    nested = re.findall(r'\(\s*SELECT\b', sql_no_strings)
    return len(nested)

def get_recursive_bool(sql):
    """
    Check if the SQL query is recursive (e.g., uses WITH RECURSIVE).
    Returns 1 if true, 0 otherwise.
    """
    sql_no_strings = strip_strings(sql)
    return int(bool(re.search(r'\bWITH\s+RECURSIVE\b', sql_no_strings)))

def get_sql_concepts_count(sql,normalized = False):
    """
    Count the number of SQL keywords present in the query.
    You can adjust the list of keywords as needed.
    """
    sql_no_strings = strip_strings(sql)
    keywords = [
        'SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT',
        'INNER JOIN', 'LEFT JOIN' , 'RIGHT JOIN', 'FULL JOIN', 'OUTER JOIN', 'ON', 'UNION',
        'EXCEPT', 'INTERSECT', 'DISTINCT', 'IN', 'EXISTS', 'NOT', 'NULL',
        'LIKE', 'BETWEEN', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'WITH'
    ] #exclude: AS,
    count = 0
    for kw in keywords:
        # Use regex to count occurrences of whole words or phrases.
        count += len(re.findall(r'\b' + re.escape(kw) + r'\b', sql_no_strings))

    if normalized:
        # Clip to avoid extremely large values
        max_len = 30
        
        # Scale from [0, max_len] -> [0, 5]
        # (clipped_length / max_len) * 5
        normalized_len = (count / float(max_len)) * 6
        return normalized_len
    else: 
        return count
    

def get_group_by_num(sql):
    """
    Count the number of columns in the GROUP BY clause.
    """
    sql_no_strings = strip_strings(sql)
    match = re.search(r'\bGROUP BY\b(.*?)(\bORDER BY\b|\bHAVING\b|\bLIMIT\b|$)', sql_no_strings, flags=re.DOTALL)
    if match:
        group_by_clause = match.group(1)
        columns = split_sql_fields(group_by_clause)
        return len(columns)
    return 0

def get_order_by_bool(sql):
    """
    Return 1 if an ORDER BY clause exists; otherwise 0.
    """
    sql_no_strings = strip_strings(sql)
    return int(bool(re.search(r'\bORDER BY\b', sql_no_strings)))

def get_limit_bool(sql):
    """
    Return 1 if a LIMIT clause exists; otherwise 0.
    """
    sql_no_strings = strip_strings(sql)
    return int(bool(re.search(r'\bLIMIT\b', sql_no_strings)))

def get_select_distinct_bool(sql):
    """
    Return 1 if SELECT DISTINCT is used; otherwise 0.
    """
    sql_no_strings = strip_strings(sql)
    return int(bool(re.search(r'\bSELECT\s+DISTINCT\b', sql_no_strings)))

def get_aggregation_count(sql):
    """
    Count how many aggregation function calls appear in the SELECT clause.
    """
    sql_no_strings = strip_strings(sql)
    match = re.search(r'\bSELECT\b(.*?)(\bFROM\b)', sql_no_strings, flags=re.DOTALL)
    agg_count = 0
    if match:
        select_clause = match.group(1)
        attributes = split_sql_fields(select_clause)
        aggregate_functions = ['SUM', 'COUNT', 'AVG', 'MIN', 'MAX']
        for attr in attributes:
            attr_count = 0
            for func in aggregate_functions:
                # Use re.findall to find *all* occurrences, not just one
                matches = re.findall(r'\b' + func + r'\s*\(', attr, flags=re.IGNORECASE)
                attr_count += len(matches)
            agg_count += attr_count
    return agg_count

def get_projected_attributes_num(sql):
    """
    Count the number of projected attributes in the SELECT clause.
    """
    sql_no_strings = strip_strings(sql)
    match = re.search(r'\bSELECT\b(.*?)(\bFROM\b)', sql_no_strings, flags=re.DOTALL)
    if match:
        select_clause = match.group(1)
        attributes = split_sql_fields(select_clause)
        return len(attributes)
    return 0

def get_num_joins(sql_str):
    """
    Count how many 'JOIN' keywords appear in a SQL query.
    This includes INNER, LEFT, RIGHT, FULL, or just JOIN.
    """
    sql_upper = sql_str.upper()
    # We look for any occurrences of 'JOIN' as a whole word
    matches = re.findall(r'\bJOIN\b', sql_upper)
    return len(matches)

def get_NLquery_len_normalized(nl_query, max_len=30, normalized = True):
    """
    Compute a normalized NL query length between 0 and 5.
    - Count words in the NL query
    - Clip at 'max_len'
    - Scale to range [0, 30]
    """
    words = nl_query.split()
    raw_length = len(words)
    if normalized:
        # Clip to avoid extremely large values
        clipped_length = min(raw_length, max_len)
        
        # Scale from [0, max_len] -> [0, 5]
        # (clipped_length / max_len) * 5
        normalized_len = (clipped_length / float(max_len)) * 6
        return normalized_len
    else: 
        return raw_length
    


def get_num_logical_operators(sql_str):
    """
    Count the occurrences of logical operators (AND, OR, NOT) in a SQL query.
    """
    sql_upper = sql_str.upper()
    count_and = len(re.findall(r'\bAND\b', sql_upper))
    count_or  = len(re.findall(r'\bOR\b', sql_upper))
    count_not = len(re.findall(r'\bNOT\b', sql_upper))
    
    return count_and + count_or + count_not

def get_num_having_conditions(sql_str):
    """
    Count the number of 'HAVING' clauses in a SQL query.
    """
    sql_upper = sql_str.upper()
    matches = re.findall(r'\bHAVING\b', sql_upper)
    return len(matches)


def get_bool_uses_like(sql_str):
    """
    Check if the SQL query uses 'LIKE'.
    Returns 1 if LIKE is present, otherwise 0.
    """
    sql_upper = sql_str.upper()
    return 1 if re.search(r'\bLIKE\b', sql_upper) else 0




def get_num_involved_columns(sql_str,normalized=False):
    """
    Count the number of unique column names in an SQL query.
    Excludes SQL keywords and table names.
    """
    sql_upper = sql_str.upper()

    # Define a set of single-word and multi-word SQL keywords
    sql_keywords = {
        'SELECT', 'FROM', 'WHERE', 'HAVING', 'LIMIT', 'ON', 'UNION',
        'EXCEPT', 'INTERSECT', 'DISTINCT', 'AS', 'IN', 'EXISTS', 'NOT', 'NULL',
        'LIKE', 'BETWEEN', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'WITH', 'AND'
    }
    
    multi_word_keywords = [
        'ORDER BY', 'GROUP BY', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL JOIN', 'OUTER JOIN'
    ]

    # Remove multi-word SQL keywords from the query
    for keyword in multi_word_keywords:
        sql_upper = sql_upper.replace(keyword, '')  # Remove as a whole phrase

    # Extract potential column names
    potential_names = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', sql_upper)

    # Extract table names (words after FROM, JOIN)
    table_names = set(re.findall(r'\bFROM\s+(\w+)|\bJOIN\s+(\w+)', sql_upper))
    table_names = {name for sublist in table_names for name in sublist if name}

    # Remove SQL keywords and table names
    unique_columns = set(col for col in potential_names if col not in sql_keywords and col not in table_names)

    if normalized:
        max_len = 30
        # Clip to avoid extremely large values
        clipped_length = min(len(unique_columns), max_len)
        
        # Scale from [0, max_len] -> [0, 5]
        # (clipped_length / max_len) * 5
        normalized_len = (clipped_length / float(max_len)) * 6
        return normalized_len
    else: 
        return len(unique_columns)

    # return len(unique_columns)
### Example usage ###
if __name__ == "__main__":
    NL_query = "Give the code of the airport with the least flights."
    sql_query = "SELECT T1.AirportCode FROM AIRPORTS AS T1 JOIN FLIGHTS AS T2 ON T1.AirportCode  =  T2.DestAirport OR T1.AirportCode  =  T2.SourceAirport GROUP BY T1.AirportCode ORDER BY count(*) LIMIT 1"

    features = extract_features(sql_query,NL_query)
    print(features)