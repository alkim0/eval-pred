use super::config::BLOCK_SIZE;
use byteorder::{NativeEndian, ReadBytesExt};
use chrono::{DateTime, Duration, NaiveDate, NaiveDateTime, Utc};
use std::alloc::{self, Layout};
use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::marker::PhantomData;
use std::mem;
use std::path::Path;
use std::slice;

pub fn sql_pattern_to_regex(s: &str) -> String {
    let mut ret = String::new();
    for c in s.chars() {
        match c {
            '_' => {
                ret.push('.');
            }
            '%' => {
                ret.push_str(".*");
            }
            _ => {
                ret.push(c);
            }
        }
    }
    ret
}

pub fn parse_duration(s: &str) -> Duration {
    let tokens: Vec<&str> = s.split_whitespace().collect();
    if tokens.len() == 1 {
        assert!(tokens[0].contains(":"));
        let mut it = tokens[0].split(":");
        let hour = it
            .next()
            .unwrap()
            .parse::<i64>()
            .expect(&format!("Could not parse hour portion of {}", tokens[0]));
        let min = it
            .next()
            .unwrap()
            .parse::<i64>()
            .expect(&format!("Could not parse minute portion of {}", tokens[0]));

        Duration::minutes(hour * 60 + min)
    } else {
        assert_eq!(tokens.len(), 2);
        let num = tokens[0]
            .parse()
            .expect(&format!("Could not parse {} {}", tokens[0], tokens[1]));
        let unit = tokens[1];
        // XXX For simplicity, 1 year is 52 weeks and 1 month is 30 days
        match unit {
            _ if unit.starts_with("year") => Duration::weeks(num * 52),
            _ if unit.starts_with("mon") => Duration::days(num * 30),
            _ if unit.starts_with("week") => Duration::weeks(num),
            _ if unit.starts_with("day") => Duration::days(num),
            _ if unit.starts_with("hour") => Duration::hours(num),
            _ if unit.starts_with("minute") => Duration::minutes(num),
            _ if unit.starts_with("second") => Duration::seconds(num),
            _ => {
                panic!("Don't support unit {}", unit);
            }
        }
    }
}

pub fn parse_datetime(s: &str) -> DateTime<Utc> {
    DateTime::from_utc(
        NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S")
            .or_else(|_| {
                NaiveDate::parse_from_str(s, "%Y%m%d").and_then(|date| Ok(date.and_hms(0, 0, 0)))
            })
            .expect(&format!("Could not parse {} as datetime", s)),
        Utc,
    )
}

// Works only on int-types
pub struct BufFile<'a, T> {
    f: File,
    file_size: usize,
    buf: &'a mut [u8],
    buf_size: usize,
    data_size: usize,
    data_per_buf: usize,
    buf_idx: usize,
    num_bufs_read: usize,
    _marker: PhantomData<T>,
}

pub trait CursorReadable {
    type T;
    fn read(rdr: &mut Cursor<&[u8]>) -> Self::T;
}

impl CursorReadable for u32 {
    type T = u32;
    fn read(rdr: &mut Cursor<&[u8]>) -> u32 {
        rdr.read_u32::<NativeEndian>().unwrap()
    }
}

impl<'a, T: CursorReadable> BufFile<'a, T> {
    pub fn new(path: &Path, buf: &'a mut [u8], buf_size: usize) -> Self {
        let mut f = File::open(path).expect(&format!("Could not open: {}", path.display()));
        let file_size = f.metadata().expect("Error finding size of file").len() as usize;

        let data_size = mem::size_of::<T>();
        let data_per_buf = buf_size / data_size;
        let bytes_read = f.read(buf).expect(&format!(
            "trouble with first read, file_size: {}",
            file_size,
        ));
        assert_eq!(bytes_read, usize::min(buf_size, file_size));

        BufFile {
            f,
            file_size,
            buf,
            buf_size,
            data_size,
            data_per_buf,
            buf_idx: 0,
            num_bufs_read: 1,
            _marker: PhantomData,
        }
    }

    pub fn get(&mut self, idx: usize) -> <T as CursorReadable>::T {
        if idx >= self.data_per_buf * (self.buf_idx + 1) {
            // If the next element is on a page later than the next page
            if idx >= self.data_per_buf * (self.buf_idx + 2) {
                self.buf_idx = idx / self.data_per_buf;
                self.f
                    .seek(SeekFrom::Start(self.buf_idx as u64 * self.buf_size as u64))
                    .expect(&format!(
                        "seeking to {} failed",
                        self.buf_idx as u64 * self.buf_size as u64
                    ));
            } else {
                self.buf_idx += 1;
            }
            let expected_to_read = usize::min((self.buf_idx + 1) * self.buf_size, self.file_size)
                - (self.buf_idx * self.buf_size);
            let bytes_read = self.f.read(self.buf).expect(&format!(
                "trouble reading from file at index {} for {} bytes",
                self.buf_idx * self.buf_size,
                expected_to_read
            ));
            assert_eq!(bytes_read, expected_to_read);
            self.num_bufs_read += 1;
        }

        let start = (idx % self.data_per_buf) * self.data_size;
        let mut rdr = Cursor::new(&self.buf[start..(start + self.data_size)]);
        T::read(&mut rdr)
    }

    pub fn num_data(&self) -> usize {
        self.file_size / self.data_size
    }

    pub fn num_bufs_read(&self) -> usize {
        self.num_bufs_read
    }

    //fn read_cursor<T>(rdr: &Cursor<&[u8]>) -> u32 {
    //    rdr.read_u32::<NativeEndian>().unwrap()
    //}
}

pub unsafe fn alloc_aligned_buf<'a>(buf_size: usize) -> &'a mut [u8] {
    slice::from_raw_parts_mut(
        alloc::alloc(
            Layout::from_size_align(buf_size, BLOCK_SIZE).expect(&format!(
                "Error with alignment settings, size: {} align: {}",
                buf_size, BLOCK_SIZE
            )),
        ),
        buf_size,
    )
}

pub unsafe fn dealloc_aligned_buf(buf: &mut [u8], buf_size: usize) {
    alloc::dealloc(
        buf.as_mut_ptr(),
        Layout::from_size_align(buf_size, BLOCK_SIZE).expect(&format!(
            "Error with alignment settings, size: {} align: {}",
            buf_size, BLOCK_SIZE
        )),
    );
}
