def process_email_body_simple(body):
    """
    Process the email body by splitting it into lines, and then splitting each line into words.
    Handles cases where the body is not a string. No quoting of words.
    """
    if not isinstance(body, str):
        return []

    # Split the email body into lines
    lines = body.split("\n")

    # Split each line into words and store them in a list
    processed_lines = [line.split() for line in lines if line.strip() != ""]

    return processed_lines 
