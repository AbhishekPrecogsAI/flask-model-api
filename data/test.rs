use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};

fn main() -> io::Result<()> {
    let file_path = "example.txt";

    // Create and write to the file
    let mut file = File::create(file_path)?;
    writeln!(file, "Line 1: Hello, Rust!")?;
    writeln!(file, "Line 2: File I/O example.")?;
    writeln!(file, "Line 3: Testing CodeChunker.")?;

    // Read the file line by line
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    println!("Reading contents of '{}':", file_path);
    for line in reader.lines() {
        println!("{}", line?);
    }

    // Append a new line to the file
    let mut file = OpenOptions::new().append(true).open(file_path)?;
    writeln!(file, "Line 4: Appending a new line.")?;

    Ok(())
}
