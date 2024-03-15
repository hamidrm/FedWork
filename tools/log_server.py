import tkinter as tk
import socket
import threading
import pickle
from tkinter import filedialog

class LogViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("FedWork Log Viewer")
        self.root.resizable(False, False) 

        self.server_ip_label = tk.Label(root, text="Server IP:")
        self.server_ip_label.grid(row=0, column=0)

        self.server_ip_entry = tk.Entry(root)
        self.server_ip_entry.grid(row=0, column=1)
        self.server_ip_entry.insert(0, "127.0.0.1")  # Default server IP

        self.server_port_label = tk.Label(root, text="Server Port:")
        self.server_port_label.grid(row=1, column=0)

        self.server_port_entry = tk.Entry(root)
        self.server_port_entry.grid(row=1, column=1)
        self.server_port_entry.insert(0, "6650")  # Default server port

        self.start_button = tk.Button(root, text="Start", command=self.start_receiving)
        self.start_button.grid(row=2, column=0)

        self.stop_button = tk.Button(root, text="Stop", command=self.stop_receiving, state=tk.DISABLED)
        self.stop_button.grid(row=2, column=1)

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_log)
        self.clear_button.grid(row=2, column=2)

        self.save_button = tk.Button(root, text="Save", command=self.save_log)
        self.save_button.grid(row=2, column=3)

        self.log_text_frame = tk.Frame(root)
        self.log_text_frame.grid(row=3, columnspan=4, sticky="nsew")
        self.log_text_frame.grid_rowconfigure(0, weight=1)
        self.log_text_frame.grid_columnconfigure(0, weight=1)

        self.log_text = tk.Text(self.log_text_frame, wrap=tk.WORD, width=80, height=50, takefocus=False)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.scrollbar_x = tk.Scrollbar(self.log_text_frame, orient="horizontal", command=self.log_text.xview)
        self.scrollbar_x.grid(row=1, column=0, sticky="ew")
        self.scrollbar_y = tk.Scrollbar(self.log_text_frame, orient="vertical", command=self.log_text.yview)
        self.scrollbar_y.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(bg="black")
        self.log_text.tag_config("color_1", foreground="white")
        self.log_text.tag_config("color_2", foreground="blue")
        self.log_text.tag_config("color_4", foreground="green")
        self.log_text.tag_config("color_8", foreground="orange")
        self.log_text.tag_config("color_16", foreground="red")
        self.log_text.configure(xscrollcommand=self.scrollbar_x.set, yscrollcommand=self.scrollbar_y.set)
        self.log_text.config(state=tk.NORMAL)

        self.server_socket = None
        self.receive_thread = None
        self.running = False

    def clear_log(self):
        self.log_text.delete("1.0", tk.END)

    def save_log(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".log",
                                              filetypes=[("Log files", "*.log"), ("All files", "*.*")])
        if file_path:
            with open(file_path, "w") as file:
                text_content = self.log_text.get("1.0", tk.END)
                file.write(text_content)

    def start_receiving(self):
        self.start_button.config(state=tk.DISABLED)
        self.server_port_entry.config(state=tk.DISABLED)
        self.server_ip_entry.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.running = True
        self.receive_thread = threading.Thread(target=self.receive_logs)
        self.receive_thread.start()

    def stop_receiving(self):
        self.running = False

    def receive_logs(self):
        server_ip = self.server_ip_entry.get()
        server_port = int(self.server_port_entry.get())
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_socket.settimeout(2.0)
        self.server_socket.bind((server_ip, server_port))

        while self.running:
            try:
                log_message_bin, _ = self.server_socket.recvfrom(1024)
            except socket.timeout:
                continue
            log_message = pickle.loads(log_message_bin)

            msg = log_message["msg"]
            log_type = log_message["type"]

            self.log_text.insert(tk.END, msg + "\n", "color_" + str(log_type))
            self.log_text.see(tk.END)

        self.server_port_entry.config(state=tk.NORMAL)
        self.server_ip_entry.config(state=tk.NORMAL)
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def on_closing(self):
        if self.server_socket:
            self.server_socket.close()
        if self.receive_thread and self.receive_thread.is_alive():
            self.running = False
            self.receive_thread.join()
        self.root.destroy()

def main():
    root = tk.Tk()
    log_viewer = LogViewer(root)
    root.protocol("WM_DELETE_WINDOW", log_viewer.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
