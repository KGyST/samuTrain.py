#!/usr/bin/env python3
"""
samuTrain Database Inspector Tool
Useful for debugging database contents and checking for issues
"""

import sqlite3
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional

class DatabaseInspector:
    def __init__(self, db_path: str = "samu.db"):
        self.db_path = db_path
        self.conn = None
        
    def connect(self) -> bool:
        """Connect to database"""
        if not os.path.exists(self.db_path):
            print(f"❌ Database not found: {self.db_path}")
            return False
        
        try:
            self.conn = sqlite3.connect(self.db_path)
            return True
        except Exception as e:
            print(f"❌ Failed to connect to database: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def get_overview(self) -> Dict[str, Any]:
        """Get database overview statistics"""
        cursor = self.conn.cursor()
        
        # Total cases
        cursor.execute('SELECT COUNT(*) FROM cases')
        total_cases = cursor.fetchone()[0]
        
        # Cases with GT text
        cursor.execute('SELECT COUNT(*) FROM cases WHERE gt_text IS NOT NULL AND gt_text != ""')
        cases_with_gt = cursor.fetchone()[0]
        
        # Cases with non-zero confidence
        cursor.execute('SELECT COUNT(*) FROM cases WHERE confidence > 0.0')
        non_zero_conf = cursor.fetchone()[0]
        
        # Cases with zero confidence
        cursor.execute('SELECT COUNT(*) FROM cases WHERE confidence = 0.0')
        zero_conf = cursor.fetchone()[0]
        
        # Failset cases
        cursor.execute('SELECT COUNT(*) FROM cases WHERE is_failset = 1')
        failset_cases = cursor.fetchone()[0]
        
        # Corrected cases
        cursor.execute('SELECT COUNT(*) FROM cases WHERE is_corrected = 1')
        corrected_cases = cursor.fetchone()[0]
        
        # Error cases
        cursor.execute('SELECT COUNT(*) FROM cases WHERE ocr_text = "ERROR"')
        error_cases = cursor.fetchone()[0]
        
        return {
            'total_cases': total_cases,
            'cases_with_gt': cases_with_gt,
            'non_zero_confidence': non_zero_conf,
            'zero_confidence': zero_conf,
            'failset_cases': failset_cases,
            'corrected_cases': corrected_cases,
            'error_cases': error_cases
        }
    
    def get_confidence_distribution(self) -> List[tuple]:
        """Get confidence distribution"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT confidence, COUNT(*) FROM cases GROUP BY confidence ORDER BY confidence')
        return cursor.fetchall()
    
    def get_sample_cases(self, limit: int = 5, confidence_filter: Optional[str] = None) -> List[Dict]:
        """Get sample cases with optional confidence filter"""
        cursor = self.conn.cursor()
        
        query = 'SELECT id, img_path, ocr_text, gt_text, confidence, is_failset, is_corrected FROM cases'
        params = []
        
        if confidence_filter:
            if confidence_filter == 'zero':
                query += ' WHERE confidence = 0.0'
            elif confidence_filter == 'nonzero':
                query += ' WHERE confidence > 0.0'
            elif confidence_filter == 'error':
                query += ' WHERE ocr_text = "ERROR"'
        
        query += ' ORDER BY id LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        cases = []
        for row in rows:
            cases.append({
                'id': row[0],
                'img_path': row[1],
                'ocr_text': row[2],
                'gt_text': row[3],
                'confidence': row[4],
                'is_failset': row[5],
                'is_corrected': row[6]
            })
        
        return cases
    
    def find_problematic_cases(self) -> List[Dict]:
        """Find cases that might indicate problems"""
        cursor = self.conn.cursor()
        problems = []
        
        # Cases with zero confidence but GT text
        cursor.execute('''
            SELECT id, img_path, ocr_text, gt_text, confidence 
            FROM cases 
            WHERE confidence = 0.0 AND gt_text IS NOT NULL AND gt_text != ""
            LIMIT 10
        ''')
        for row in cursor.fetchall():
            problems.append({
                'type': 'zero_confidence_with_gt',
                'id': row[0],
                'img_path': row[1],
                'ocr_text': row[2],
                'gt_text': row[3],
                'confidence': row[4]
            })
        
        # Cases with OCR errors
        cursor.execute('''
            SELECT id, img_path, ocr_text, gt_text, confidence 
            FROM cases 
            WHERE ocr_text = "ERROR"
            LIMIT 10
        ''')
        for row in cursor.fetchall():
            problems.append({
                'type': 'ocr_error',
                'id': row[0],
                'img_path': row[1],
                'ocr_text': row[2],
                'gt_text': row[3],
                'confidence': row[4]
            })
        
        return problems
    
    def search_cases(self, search_term: str, field: str = 'all') -> List[Dict]:
        """Search cases by text in various fields"""
        cursor = self.conn.cursor()
        
        if field == 'all':
            cursor.execute('''
                SELECT id, img_path, ocr_text, gt_text, confidence 
                FROM cases 
                WHERE ocr_text LIKE ? OR gt_text LIKE ? OR img_path LIKE ?
                LIMIT 20
            ''', (f'%{search_term}%', f'%{search_term}%', f'%{search_term}%'))
        elif field == 'ocr':
            cursor.execute('''
                SELECT id, img_path, ocr_text, gt_text, confidence 
                FROM cases 
                WHERE ocr_text LIKE ?
                LIMIT 20
            ''', (f'%{search_term}%',))
        elif field == 'gt':
            cursor.execute('''
                SELECT id, img_path, ocr_text, gt_text, confidence 
                FROM cases 
                WHERE gt_text LIKE ?
                LIMIT 20
            ''', (f'%{search_term}%',))
        elif field == 'path':
            cursor.execute('''
                SELECT id, img_path, ocr_text, gt_text, confidence 
                FROM cases 
                WHERE img_path LIKE ?
                LIMIT 20
            ''', (f'%{search_term}%',))
        
        rows = cursor.fetchall()
        return [{
            'id': row[0],
            'img_path': row[1],
            'ocr_text': row[2],
            'gt_text': row[3],
            'confidence': row[4]
        } for row in rows]

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Inspect samuTrain database')
    parser.add_argument('--db', default='samu.db', help='Database file path')
    parser.add_argument('--overview', action='store_true', help='Show database overview')
    parser.add_argument('--confidence', action='store_true', help='Show confidence distribution')
    parser.add_argument('--sample', type=int, default=5, help='Show sample cases')
    parser.add_argument('--filter', choices=['zero', 'nonzero', 'error'], help='Filter cases by confidence')
    parser.add_argument('--problems', action='store_true', help='Find problematic cases')
    parser.add_argument('--search', help='Search term')
    parser.add_argument('--field', choices=['all', 'ocr', 'gt', 'path'], default='all', help='Search field')
    
    args = parser.parse_args()
    
    inspector = DatabaseInspector(args.db)
    if not inspector.connect():
        sys.exit(1)
    
    try:
        if args.overview:
            overview = inspector.get_overview()
            print("📊 Database Overview:")
            print(f"  Total cases: {overview['total_cases']}")
            print(f"  Cases with GT text: {overview['cases_with_gt']}")
            print(f"  Cases with confidence > 0.0: {overview['non_zero_confidence']}")
            print(f"  Cases with confidence = 0.0: {overview['zero_confidence']}")
            print(f"  Failset cases: {overview['failset_cases']}")
            print(f"  Corrected cases: {overview['corrected_cases']}")
            print(f"  Error cases: {overview['error_cases']}")
            print()
        
        if args.confidence:
            dist = inspector.get_confidence_distribution()
            print("📈 Confidence Distribution:")
            for conf, count in dist:
                print(f"  {conf}: {count} cases")
            print()
        
        if args.problems:
            problems = inspector.find_problematic_cases()
            if problems:
                print("⚠️ Problematic Cases:")
                for prob in problems:
                    print(f"  {prob['type'].upper()}: ID {prob['id']}, Path: {prob['img_path']}")
                    print(f"    OCR: '{prob['ocr_text']}'")
                    print(f"    GT: '{prob['gt_text']}'")
                    print(f"    Conf: {prob['confidence']}")
                    print()
            else:
                print("✅ No problematic cases found")
                print()
        
        if args.search:
            results = inspector.search_cases(args.search, args.field)
            print(f"🔍 Search results for '{args.search}' in {args.field}:")
            for result in results:
                print(f"  ID: {result['id']}, Path: {result['img_path']}")
                print(f"    OCR: '{result['ocr_text']}'")
                print(f"    GT: '{result['gt_text']}'")
                print(f"    Conf: {result['confidence']}")
                print()
        
        if args.sample > 0:
            filter_arg = args.filter if hasattr(args, 'filter') else None
            cases = inspector.get_sample_cases(args.sample, filter_arg)
            filter_desc = f" (filter: {filter_arg})" if filter_arg else ""
            print(f"📋 Sample cases{filter_desc}:")
            for case in cases:
                print(f"  ID: {case['id']}, Path: {case['img_path']}")
                print(f"    OCR: '{case['ocr_text']}'")
                print(f"    GT: '{case['gt_text']}'")
                print(f"    Conf: {case['confidence']}")
                print(f"    Failset: {case['is_failset']}, Corrected: {case['is_corrected']}")
                print()
        
        # Default: show overview
        if not any([args.overview, args.confidence, args.problems, args.search, args.sample > 0]):
            overview = inspector.get_overview()
            print("📊 Database Overview:")
            print(f"  Total cases: {overview['total_cases']}")
            print(f"  Cases with GT text: {overview['cases_with_gt']}")
            print(f"  Cases with confidence > 0.0: {overview['non_zero_confidence']}")
            print(f"  Cases with confidence = 0.0: {overview['zero_confidence']}")
            print(f"  Failset cases: {overview['failset_cases']}")
            print(f"  Corrected cases: {overview['corrected_cases']}")
            print(f"  Error cases: {overview['error_cases']}")
            
    finally:
        inspector.close()

if __name__ == "__main__":
    main()
