#!/usr/bin/env python3
"""
Script to print database cases table content.
Prints id, model_prediction (pred), gt_text (gt), and confidence fields.
Ordered by confidence in descending order.
"""

import sqlite3
import sys
import argparse


def print_db_cases(db_path: str = "samu.db", limit: int = None):
    """Print cases from database with id, pred, gt, and confidence ordered by confidence desc"""
    
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Build query
            query = """
                SELECT id, model_prediction as pred, gt_text as gt, confidence, img_path
                FROM cases
                ORDER BY confidence DESC
            """
            
            params = []
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            if not rows:
                print(f"No cases found in database: {db_path}")
                return
            
            # Print header
            print(f"Database: {db_path}")
            print(f"Total cases: {len(rows)}")
            print("-" * 110)
            print(f"{'ID':<6} {'BASE':<20} {'PRED':<25} {'GT':<25} {'CONF':<8}")
            print("-" * 110)
            
            # Print each row
            for row in rows:
                id_val = row['id']
                pred = row['pred'] or ""
                gt = row['gt'] or ""
                conf = row['confidence']
                img_path = row['img_path']
                
                # Extract base filename (remove extensions)
                import os
                base_name = os.path.basename(img_path)
                if base_name.endswith('.bin.png'):
                    base_name = base_name[:-8]  # Remove .bin.png
                elif base_name.endswith('.png'):
                    base_name = base_name[:-4]   # Remove .png
                elif '.' in base_name:
                    base_name = base_name.rsplit('.', 1)[0]  # Remove last extension
                
                # Truncate long strings for display
                pred_display = pred[:22] + "..." if len(pred) > 25 else pred
                gt_display = gt[:22] + "..." if len(gt) > 25 else gt
                base_display = base_name[:17] + "..." if len(base_name) > 20 else base_name
                
                print(f"{id_val:<6} {base_display:<20} {pred_display:<25} {gt_display:<25} {conf:<8.3f}")
            
            print("-" * 110)
            
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Print database cases table content")
    parser.add_argument("--db", default="samu.db", help="Database file path (default: samu.db)")
    parser.add_argument("--limit", type=int, help="Limit number of rows to display")
    
    args = parser.parse_args()
    
    print_db_cases(args.db, args.limit)


if __name__ == "__main__":
    main()
