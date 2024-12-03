import sqlite3
from datetime import datetime

class AttendanceManager:
    def __init__(self, db_path='attendance.db'):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        # Create the attendance table if it doesn't exist
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            student_id TEXT,
            name TEXT,
            date TEXT,
            time TEXT,
            PRIMARY KEY (student_id, date)
        )
        ''')
        self.conn.commit()

    def mark_attendance(self, student_id, name):
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H:%M:%S')

        # Check if attendance is already marked for the student on this date
        self.cursor.execute('SELECT * FROM attendance WHERE student_id = ? AND date = ?', (student_id, current_date))
        data = self.cursor.fetchone()

        if data is None:
            # Insert attendance record
            self.cursor.execute('INSERT INTO attendance (student_id, name, date, time) VALUES (?, ?, ?, ?)',
                                (student_id, name, current_date, current_time))
            self.conn.commit()
            print(f"Attendance marked for {name} at {current_time}.")
        else:
            print(f"Attendance already marked for {name}.")

    def close(self):
        self.conn.close()
