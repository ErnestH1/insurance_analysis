**Prerequisites:**
1. Python installed on your machine.
2. Pip installed (comes with Python by default).
3. Use and IDE for example Visual Studio Code

**Setup Instructions:**

1. **Clone or Download the API Project:**
   Clone or download the API project.
   Then save it to a preferred location on your machine.

2. **Navigate to the Project Directory:**
   Open a terminal or command prompt and navigate to the directory where the API project is saved using the `cd` command. For example:

        `cd C:\Users\Admin\Api`


3. **Create and Activate Virtual Environment (Optional but Recommended):**
It's a good practice to create a virtual environment to isolate the project dependencies. Users can create a virtual environment using the following command:

        `python -m venv myenv`

Then, they can activate the virtual environment:
- On Windows:
  ```
  myenv\Scripts\activate
  ```
- On macOS and Linux:
  ```
  source myenv/bin/activate
  ```

4. **Install Required Dependencies:**
Users should install the required Python dependencies listed in the `requirements.txt` file using pip:

        `pip install -r requirements.txt`

5. **Run the API using Uvicorn:**
To run the API using Uvicorn, users can execute the following command in the Terminal:

        `uvicorn app:app --reload`

- `app:app` specifies the location of the FastAPI application object.
- `--reload` enables automatic reloading of the server when code changes are detected, which is helpful during development.

6. **Accessing the API:**
Once the API server is running, users can access it using a web browser or tools like Postman. By default, the API will be accessible at `http://localhost:8000`.

7. **Making Predictions:**
Users can make predictions by sending HTTP requests to the API endpoints defined in the FastAPI application. 




