#!/usr/bin/env python3
"""
Colonial Pipeline Transit Times Data Extractor

Extracts transit time data from Outlook emails containing Colonial Pipeline bulletins.
Supports configurable From/To location parameters.
"""

import re
import warnings
import pandas as pd
import win32com.client
from io import StringIO
from datetime import timedelta
from dateutil import parser
from bs4 import BeautifulSoup
from typing import Optional, Tuple, List, Dict, Any


class ColonialTransitExtractor:
    """Extracts Colonial Pipeline transit time data from Outlook emails."""
    
    def __init__(self, target_subject: str = "T4 Bulletin: Colonial - TRANSIT TIMES sent to sto_susan"):
        """
        Initialize the extractor.
        
        Args:
            target_subject: Email subject line to search for
        """
        self.target_subject = target_subject
    
    def _create_soup(self, html: str) -> BeautifulSoup:
        """Create BeautifulSoup object with fallback parsers."""
        try:
            return BeautifulSoup(html, "lxml")
        except:
            return BeautifulSoup(html, "html.parser")
    
    def _extract_date_from_text(self, text: str, fallback_date) -> Any:
        """Extract date from email text with fallback."""
        try:
            if "Date:" in text:
                date_part = text.split("Date:", 1)[1].split("\n", 1)[0]
                return parser.parse(date_part, fuzzy=True).date() - timedelta(days=1)
            return parser.parse(text, fuzzy=True).date() - timedelta(days=1)
        except Exception:
            return fallback_date - timedelta(days=1)
    
    def _promote_header_row(self, df: pd.DataFrame) -> pd.DataFrame:
        """Find and promote the row containing 'From' and 'To' as column headers."""
        for i in range(min(8, len(df))):
            values = [str(v).strip().lower() for v in df.iloc[i].values]
            if "from" in values and "to" in values:
                df.columns = df.iloc[i].astype(str).str.strip()
                return df.iloc[i+1:].reset_index(drop=True)
        return df
    
    def _normalize_location_code(self, code: str) -> str:
        """Normalize location code by keeping only uppercase letters."""
        return re.sub(r"[^A-Z]", "", str(code).upper())
    
    def _extract_first_four_numbers(self, df: pd.DataFrame, row_idx: int, start_col_idx: int) -> List[Any]:
        """
        Extract the first four numeric values from a row starting from start_col_idx+1.
        
        Returns:
            List of four values: [gas_days, gas_hours, distillates_days, distillates_hours]
        """
        numbers = []
        for value in df.iloc[row_idx, start_col_idx+1:].tolist():
            try:
                numeric_val = pd.to_numeric(str(value).strip(), errors="coerce")
            except Exception:
                numeric_val = pd.NA
            
            if pd.notna(numeric_val) and float(numeric_val).is_integer():
                numbers.append(int(numeric_val))
            
            if len(numbers) >= 4:  # Gas Days, Gas Hours, Distillates Days, Distillates Hours
                break
        
        # Pad with NA values if we don't have 4 numbers
        while len(numbers) < 4:
            numbers.append(pd.NA)
        
        return numbers
    
    def _find_column_indices(self, df: pd.DataFrame) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Find column indices for From, To, and Cycle columns."""
        columns = [str(c).strip() for c in df.columns]
        lowercase_cols = [c.lower() for c in columns]
        
        try:
            from_idx = next(i for i, c in enumerate(lowercase_cols) if c == "from")
        except StopIteration:
            from_idx = None
        
        try:
            to_idx = next(i for i, c in enumerate(lowercase_cols) if c == "to")
        except StopIteration:
            to_idx = None
        
        try:
            cycle_idx = next((i for i, c in enumerate(lowercase_cols) if "cycle" in c), None)
        except StopIteration:
            cycle_idx = None
        
        return from_idx, to_idx, cycle_idx
    
    def _extract_cycle_value(self, df: pd.DataFrame, row_idx: int, cycle_col_idx: Optional[int]) -> Any:
        """Extract and validate cycle value from the specified column."""
        if cycle_col_idx is None:
            return pd.NA
        
        try:
            cycle_val = pd.to_numeric(df.iat[row_idx, cycle_col_idx], errors="coerce")
            if pd.notna(cycle_val) and float(cycle_val).is_integer():
                return int(cycle_val)
        except Exception:
            pass
        
        return pd.NA
    
    def extract_transit_data(self, from_location: str = "HTN", to_location: str = "GBJ") -> pd.DataFrame:
        """
        Extract Colonial Pipeline transit time data from Outlook emails.
        
        Args:
            from_location: Source location code (default: "HTN")
            to_location: Destination location code (default: "GBJ")
        
        Returns:
            DataFrame with columns: Date, From, To, Cycle, Gas Days, Gas Hours, 
                                  Distillates Days, Distillates Hours
        """
        # Normalize location codes
        from_code = self._normalize_location_code(from_location)
        to_code = self._normalize_location_code(to_location)
        
        # Connect to Outlook
        try:
            outlook = win32com.client.Dispatch("Outlook.Application")
            namespace = outlook.GetNamespace("MAPI")
            inbox = namespace.GetDefaultFolder(6)  # Inbox folder
            items = inbox.Items
            items.Sort("[ReceivedTime]", True)  # Sort by received time, descending
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Outlook: {e}")
        
        extracted_data = []
        
        # Process each email
        for message in items:
            subject = getattr(message, "Subject", "") or ""
            if self.target_subject not in subject:
                continue
            
            html_body = getattr(message, "HTMLBody", "") or ""
            if not html_body:
                continue
            
            # Parse HTML and extract date
            soup = self._create_soup(html_body)
            text_content = soup.get_text("\n", strip=True)
            email_date = self._extract_date_from_text(text_content, message.ReceivedTime.date())
            
            # Process all tables in the email
            for table in soup.find_all("table"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    try:
                        dataframes = pd.read_html(StringIO(str(table)))
                    except Exception:
                        continue
                
                # Process each DataFrame from the table
                for df in dataframes:
                    if df.empty:
                        continue
                    
                    # Promote header row
                    df = self._promote_header_row(df)
                    
                    # Find column indices
                    from_idx, to_idx, cycle_idx = self._find_column_indices(df)
                    if from_idx is None or to_idx is None:
                        continue
                    
                    # Clean location codes
                    df["From"] = df.iloc[:, from_idx].map(self._normalize_location_code)
                    df["To"] = df.iloc[:, to_idx].map(self._normalize_location_code)
                    
                    # Determine starting column for number extraction
                    start_col_idx = cycle_idx if cycle_idx is not None else max(from_idx, to_idx)
                    
                    # Filter rows matching the specified route
                    route_mask = (df["From"] == from_code) & (df["To"] == to_code)
                    matching_rows = df.index[route_mask]
                    
                    # Extract data from matching rows
                    for row_idx in matching_rows:
                        gas_days, gas_hours, dist_days, dist_hours = self._extract_first_four_numbers(
                            df, row_idx, start_col_idx
                        )
                        
                        cycle_value = self._extract_cycle_value(df, row_idx, cycle_idx)
                        
                        extracted_data.append({
                            "Date": email_date,
                            "From": from_code,
                            "To": to_code,
                            "Cycle": cycle_value,
                            "Gas Days": gas_days,
                            "Gas Hours": gas_hours,
                            "Distillates Days": dist_days,
                            "Distillates Hours": dist_hours
                        })
        
        # Create final DataFrame
        if not extracted_data:
            return pd.DataFrame(columns=[
                "Date", "From", "To", "Cycle", "Gas Days", "Gas Hours", 
                "Distillates Days", "Distillates Hours"
            ])
        
        result_df = (pd.DataFrame(extracted_data)
                    .sort_values("Date", ascending=False)
                    .reset_index(drop=True))
        
        return result_df


def extract_colonial_transit_times(from_location: str = "HTN", 
                                 to_location: str = "GBJ",
                                 target_subject: str = "T4 Bulletin: Colonial - TRANSIT TIMES sent to sto_susan") -> pd.DataFrame:
    """
    Convenience function to extract Colonial Pipeline transit time data.
    
    Args:
        from_location: Source location code (default: "HTN")
        to_location: Destination location code (default: "GBJ")
        target_subject: Email subject line to search for
    
    Returns:
        DataFrame with transit time data
    """
    extractor = ColonialTransitExtractor(target_subject)
    return extractor.extract_transit_data(from_location, to_location)


def main():
    """Main function for command-line usage."""
    # Example usage with default parameters (HTN -> GBJ)
    print("Extracting Colonial Pipeline transit times (HTN -> GBJ)...")
    df = extract_colonial_transit_times()
    
    if not df.empty:
        print(f"\nExtracted {len(df)} records:")
        print(df.head())
        
        # Optionally save to Excel
        # df.to_excel("ColonialTransitTimes_HTN_to_GBJ.xlsx", index=False)
        # print("\nData saved to ColonialTransitTimes_HTN_to_GBJ.xlsx")
    else:
        print("No data found matching the criteria.")

if __name__ == "__main__":
    main()