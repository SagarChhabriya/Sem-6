
## 🛠️ Step 1: Set Up a Simple HTTP Server

We'll use Python's built-in HTTP server for this purpose. Here's how you can set it up:

1. **Create a Directory for Your Server Files**:
   - Open your terminal or command prompt.
   - Create a new directory:
     ```bash
     mkdir SimpleHTTPServer
     cd SimpleHTTPServer
     ```

2. **Create an HTML File with a Login Form**:
   - Inside the `SimpleHTTPServer` directory, create a file named `index.html` with the following content:
     ```html
     <!DOCTYPE html>
     <html>
     <head>
       <title>Login Form</title>
     </head>
     <body>
       <h2>Login</h2>
       <form action="/submit" method="POST">
         Username: <input type="text" name="username"><br><br>
         Password: <input type="password" name="password"><br><br>
         <input type="submit" value="Submit">
       </form>
     </body>
     </html>
     ```
   - This form will send the username and password as plain text over HTTP when submitted.

3. **Start the HTTP Server**:
   - In the same directory, start the server:
     ```bash
     python3 -m http.server 8080
     ```
   - This command will start a simple HTTP server on port 8080. You can access it by navigating to `http://localhost:8080` in your web browser.

---

## 🕵️‍♂️ Step 2: Sniff HTTP Traffic with Wireshark

To intercept and analyze the HTTP traffic:

1. **Install Wireshark**:
   - Download and install Wireshark from [here](https://www.wireshark.org/download.html).

2. **Start Capturing Traffic**:
   - Open Wireshark and select the network interface that your computer is using (e.g., Wi-Fi or Ethernet).
   - Start the capture by clicking the shark fin icon.

3. **Apply a Display Filter**:
   - To focus on HTTP traffic, apply the following filter:
     ```
     http
     ```
   - This will display only HTTP packets, making it easier to analyze the traffic.

4. **Submit the Login Form**:
   - Go back to your browser and navigate to `http://localhost:8080`.
   - Fill in the username and password fields and submit the form.

5. **Analyze the Captured Traffic**:
   - In Wireshark, look for the HTTP POST request to `/submit`.
   - Expand the packet details to see the form data, including the username and password sent in plain text.

---

## 🔐 Important Notes

- **Security Implications**: This demonstration highlights the risks of transmitting sensitive information over unencrypted HTTP. In real-world scenarios, always use HTTPS to encrypt data in transit.
  
- **Ethical Considerations**: Only perform network sniffing on networks and systems you own or have explicit permission to test. Unauthorized interception of network traffic is illegal and unethical.
