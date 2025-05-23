# Conceptual Tests for URL Ingestion Feature

This document outlines conceptual test cases for the web content ingestion feature. These tests are intended for manual verification or as a basis for future automated testing.

## 1. UI and Input Handling

*   **Test Case 1.1: URL Input Field Present**
    *   **Objective:** Verify the URL input field is visible and usable in the sidebar.
    *   **Steps:** Open the application.
    *   **Expected Result:** A text input field for URLs is present in the sidebar.

*   **Test Case 1.2: "Process Documents" Button State (No Input)**
    *   **Objective:** Verify the "Process Documents" button is disabled when no files are uploaded and no URL is entered.
    *   **Steps:** Open the application. Ensure no files are staged for upload and the URL input is empty.
    *   **Expected Result:** The "Process Documents" button is disabled. The info message "Please upload PDF or Word documents, or enter a URL to begin." is displayed.

*   **Test Case 1.3: "Process Documents" Button State (URL Only)**
    *   **Objective:** Verify the "Process Documents" button is enabled when a URL is entered, even with no files.
    *   **Steps:** Open the application. Enter a valid URL in the input field. Do not upload any files.
    *   **Expected Result:** The "Process Documents" button becomes enabled.

*   **Test Case 1.4: "Process Documents" Button State (Files Only)**
    *   **Objective:** Verify the "Process Documents" button is enabled when files are uploaded, even with no URL.
    *   **Steps:** Open the application. Upload a PDF or DOCX file. Ensure the URL input is empty.
    *   **Expected Result:** The "Process Documents" button becomes enabled.

*   **Test Case 1.5: Display of URL for Processing**
    *   **Objective:** Verify the entered URL is displayed correctly before processing.
    *   **Steps:** Enter a URL in the input field.
    *   **Expected Result:** The entered URL is listed under the "Items to Process" (or similar) section.

## 2. URL Fetching and Content Extraction

*   **Test Case 2.1: Valid URL - Successful Fetch and Extraction**
    *   **Objective:** Verify successful text extraction from a valid, publicly accessible URL with simple HTML content.
    *   **Steps:**
        1.  Enter a known URL with clear text content (e.g., a simple blog post or news article).
        2.  Click "Process Documents".
        3.  Ask a question related to the content of the URL.
    *   **Expected Result:** The application processes the URL, extracts text, and the chatbot can answer questions based on the URL's content. A success message is shown.

*   **Test Case 2.2: Valid URL - Complex HTML**
    *   **Objective:** Verify text extraction from a URL with a more complex HTML structure (e.g., a documentation page).
    *   **Steps:**
        1.  Enter a URL for a page with sidebars, navigation, footers, etc.
        2.  Click "Process Documents".
        3.  Ask a question related to the main content of the URL.
    *   **Expected Result:** The application extracts mainly the primary textual content, ignoring boilerplate. Chatbot answers accurately.

*   **Test Case 2.3: URL with No Meaningful Text**
    *   **Objective:** Verify behavior when a URL points to a page with no extractable text (e.g., an image gallery or a page with only JavaScript-rendered content if not handled).
    *   **Steps:**
        1.  Enter a URL known to have minimal or no direct text.
        2.  Click "Process Documents".
    *   **Expected Result:** A warning message like "Could not extract meaningful text content from the URL" or "The URL was fetched and parsed, but no text content was found" is displayed. The `url_processed_successfully` flag should be false.

## 3. Error Handling

*   **Test Case 3.1: Invalid URL Format**
    *   **Objective:** Verify error handling for malformed URLs.
    *   **Steps:**
        1.  Enter an invalid URL (e.g., "htp://example.com", "example", "www.examplecom").
        2.  Click "Process Documents".
    *   **Expected Result:** An appropriate error message is displayed (e.g., "Invalid URL format. Please include http:// or https://"). Processing should not proceed for the URL.

*   **Test Case 3.2: Non-existent URL (404 Error)**
    *   **Objective:** Verify error handling for URLs that result in a 404 error.
    *   **Steps:**
        1.  Enter a URL that is known to be non-existent (e.g., "http://example.com/nonexistentpage123").
        2.  Click "Process Documents".
    *   **Expected Result:** An error message like "URL returned a 404 Not Found error" is displayed.

*   **Test Case 3.3: Forbidden URL (403 Error)**
    *   **Objective:** Verify error handling for URLs that result in a 403 error.
    *   **Steps:**
        1.  Enter a URL that is known to return a 403 Forbidden error.
        2.  Click "Process Documents".
    *   **Expected Result:** An error message like "Access to URL denied (403 Forbidden)" is displayed.

*   **Test Case 3.4: Connection Timeout**
    *   **Objective:** Verify error handling for URLs that time out. (This might be hard to reliably reproduce manually but is important).
    *   **Steps:**
        1.  (If possible) Use a tool to simulate a very slow responding URL or enter a URL known to be problematic.
        2.  Click "Process Documents".
    *   **Expected Result:** An error message like "The request to the URL timed out" is displayed.

*   **Test Case 3.5: Connection Error (DNS Failure)**
    *   **Objective:** Verify error handling for DNS lookup failures.
    *   **Steps:**
        1.  Enter a URL with a completely invalid domain (e.g., "http://thisshouldnotexist12345.com").
        2.  Click "Process Documents".
    *   **Expected Result:** An error message like "Failed to connect to the URL. Please check the URL and your internet connection" is displayed.

## 4. Integration with RAG Pipeline

*   **Test Case 4.1: URL Content Used by Chatbot**
    *   **Objective:** Verify that text extracted from a URL is correctly chunked, vectorized, and used by the RAG pipeline.
    *   **Steps:**
        1.  Process a URL with known, unique content.
        2.  Process a separate PDF document with different known, unique content.
        3.  Ask a question that can *only* be answered by the URL's content.
        4.  Ask a question that can *only* be answered by the PDF's content.
    *   **Expected Result:** The chatbot answers both questions correctly, demonstrating that both sources were processed and are usable.

*   **Test Case 4.2: Mixed Content (URL and File)**
    *   **Objective:** Verify successful processing when both a URL and a file are provided.
    *   **Steps:**
        1.  Upload a PDF/DOCX file.
        2.  Enter a valid URL.
        3.  Click "Process Documents".
        4.  Ask questions pertaining to both the file and the URL content.
    *   **Expected Result:** The application processes both sources. Chatbot can answer questions based on content from either.

## 5. Security (Conceptual)

*   **Test Case 5.1: No Arbitrary Code Execution from URL Content**
    *   **Objective:** Conceptually verify that processing URL content does not lead to arbitrary code execution (e.g., JavaScript from the page is not executed by the backend).
    *   **Note:** This relies on `BeautifulSoup` and `requests` behaving as expected, stripping active content. Direct testing is complex.
    *   **Expected Behavior:** Only text is extracted; no scripts or active content from the webpage are executed on the server.
